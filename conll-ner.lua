require 'torch'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-train', 'data/conll2003/eng.train.torch','torch format train file list')
cmd:option('-test', 'data/conll2003/eng.testa.torch','torch format test file list')
cmd:option('-minibatch', 64,'minibatch size')
cmd:option('-cuda', 0,'whether to use gpu')
cmd:option('-labelDim', 8,'label dimension')
cmd:option('-embeddingDim', 64,'embedding dimension')
cmd:option('-vocabSize', 100004,'vocabulary size')
cmd:option('-sentenceLength', 5,'length of input sequences')
cmd:option('-learningRate', 0.01,'init learning rate')
cmd:option('-numEpochs', 5, 'number of epochs to train for')
cmd:option('-evaluateFrequency', 5, 'number of epochs to train for')
cmd:option('-preloadEmbeddings', '', 'file containing serialized torch embeddings')

local params = cmd:parse(arg)


useCuda = params.cuda == 1
if(useCuda) then
    require 'cunn'
    print('using GPU')
else
    require 'nn'
    print ('using CPU')
end
local function toCuda(x)
    local y = x
    if(useCuda) then
        y = x:cuda()
    end
    return y
end

--- data parameters
local train_file = params.train
local test_file = params.test
local sentenceLength = params.sentenceLength
local vocabSize = params.vocabSize
local train = torch.load(train_file)

--- model parameters
local embeddingDim = params.embeddingDim
local hiddenUnits = 300
local minibatchSize = params.minibatch
local numClasses = params.labelDim
local concatDim = embeddingDim * sentenceLength

--- optimization parameters
local optConfig = {
    learningRate = params.learningRate,
--    learningRateDecay = params.learningRateDecay,
--    momentum = useMomentum,
--    dampening = dampening,
}
local optState = {}
local optimMethod = optim.sgd
local numEpochs = params.numEpochs
local numBatches = math.floor(train.data:size()[1]/minibatchSize)


--- split training data into batches
local dataBatches = {}
local labelsBatches = {}
local startIdx = 1
local endIdx =  startIdx + minibatchSize - 1
while(endIdx - 1 < train.labels:size()[1])
do
    table.insert(dataBatches,toCuda(train.data:narrow(1, startIdx, endIdx-startIdx-1)))
    table.insert(labelsBatches,toCuda(train.labels:narrow(1, startIdx, endIdx-startIdx-1)))
    startIdx = endIdx
    endIdx = startIdx + minibatchSize+1
end

---- preload embeddings if specified ----
local lookupTable = nn.LookupTable(vocabSize,embeddingDim)
if params.preloadEmbeddings ~= '' then
    local data = torch.load(params.preloadEmbeddings)
    vocabSize = data.data:size()[1]
    embeddingDim = data.data:size()[2]
    lookupTable = nn.LookupTable(vocabSize,embeddingDim)
    lookupTable.weights = data.data
end

---- setup network from nlp afs ----
local net = nn.Sequential()
net:add(lookupTable)
net:add(nn.Reshape(concatDim))
net:add(nn.Linear(concatDim, hiddenUnits))
net:add(nn.HardTanh())
net:add(nn.Linear(hiddenUnits, numClasses))
net:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()
toCuda(criterion)
toCuda(net)

--- Evaluate ---
local function evaluate()
    print ('Evaluating')
    local test = torch.load(test_file)
    local tp = 0
    local fp = 0
    local tn = 0
    local fn = 0
    for i = 1, test.labels:size()[1]
    do
        local sample = toCuda(test.data:narrow(1, i, 1))
        local label = test.labels:narrow(1, i, 1)[1]
        local pred = net:forward(sample)
        local max_prob = -math.huge
        local max_index = -math.huge
        for j = 1, numClasses
        do
            if pred[1][j] > max_prob then
                max_prob = pred[1][j]
                max_index = j
            end
        end
        if label == 1 and max_index == label then tn = tn + 1 end
        if label == 1 and max_index ~= label then fp = fp + 1 end
        if label ~= 1 and max_index == label then tp = tp + 1 end
        if label ~= 1 and max_index == 1 then fn = fn + 1 end
    end

    local precision = tp / (tp + fp)
    local recall = tp / (tp + fn)
    local f1 = 2 * ((precision * recall) / (precision + recall))
    print(string.format('F1 : %f\t Recall : %f\tPrecision : %f', f1, recall, precision))
end


--- Train ---
local function train()
    local parameters, gradParameters = net:getParameters()

    for epoch = 1, numEpochs
    do
        local epoch_error = 0
        local startTime = sys.clock()
        print('Starting epoch ' .. epoch .. ' of ' .. numEpochs)
        -- TODO wtf is wrong with the end of this
        for i = 1, numBatches - 100
        do
            local idx = (i % numBatches) + 1
            local sentences = dataBatches[idx]
            local labels = labelsBatches[idx]

            -- update function
            local function fEval(x)
                if parameters ~= x then parameters:copy(x) end
                net:zeroGradParameters()
                local output = net:forward(sentences)
                local err = criterion:forward(output,labels)
                local df_do = criterion:backward(output, labels)
                net:backward(sentences, df_do)
                epoch_error = epoch_error+err
                return err, gradParameters
            end
            -- update gradients
            optimMethod(fEval, parameters, optConfig, optState)

            if(i % 500 == 0) then
                print(string.format('%f percent complete \t speed = %f examples/sec',
                    i/(numBatches), (i*minibatchSize)/(sys.clock() - startTime)))
            end
        end
        print(string.format('Epoch error = %f', epoch_error))
        if (epoch % params.evaluateFrequency == 0) then evaluate() end
    end
end

train()