require 'torch'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-train', 'data/conll2003/eng.train.torch','torch format train file list')
cmd:option('-test', 'data/conll2003/eng.testa.torch','torch format test file list')
cmd:option('-minibatch', 64,'minibatch size')
cmd:option('-cuda', 0,'whether to use gpu')
cmd:option('-labelDim', 8,'label dimension')
cmd:option('-labelMap', 'data/conll2003/lebel-map.index','file containing map from label strings to index. needed for entity level evaluation')
cmd:option('-embeddingDim', 64,'embedding dimension')
cmd:option('-vocabSize', 100004,'vocabulary size')
cmd:option('-sentenceLength', 5,'length of input sequences')
cmd:option('-learningRate', 0.01,'init learning rate')
cmd:option('-numEpochs', 5, 'number of epochs to train for')
cmd:option('-evaluateFrequency', 5, 'number of epochs to train for')
cmd:option('-loadEmbeddings', '', 'file containing serialized torch embeddings')
cmd:option('-saveModel', '', 'file to save the trained model to')

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
local label_map_file = params.labelMap
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

---- preload embeddings if specified ----
local lookupTable = nn.LookupTable(vocabSize,embeddingDim)
if params.loadEmbeddings ~= '' then
    print('preloading embeddings from ' .. params.loadEmbeddings)
    local data = torch.load(params.loadEmbeddings)
    vocabSize = data.data:size()[1]
    embeddingDim = data.data:size()[2]
    lookupTable = nn.LookupTable(vocabSize,embeddingDim)
    lookupTable.weight = data.data
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
    -- load maps to chunk entities
    local label_map = {}
    for line in io.lines(params.labelMap) do
        local label_string, label_index = string.match(line,'([^\t]+)\t([^\t]+)')
        label_map[label_index] = label_string
    end

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
    return f1
end


--- Train ---
local function train_model()
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

    local parameters, gradParameters = net:getParameters()
    local last_f1 = 0.0
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
        if (epoch % params.evaluateFrequency == 0) then
            local f1 = evaluate()
            -- end training early if f1 goes down
            -- TODO we want the last/better model, not this one
            if f1 < last_f1 then break end
        end
    end
    -- save the trained model if location specified
    if params.saveModel ~= '' then torch.save(params.saveModel, net) end
end

train_model()