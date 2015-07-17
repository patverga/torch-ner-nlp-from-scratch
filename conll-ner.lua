require 'torch'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-train', 'train','torch format train file list')
cmd:option('-test', 'testa','torch format test file list')
cmd:option('-minibatch', 32,'minibatch size')
cmd:option('-cuda', 0,'whether to use gpu')
cmd:option('-labelDim', 8,'label dimension')
cmd:option('-embeddingDim', 64,'embedding dimension')
cmd:option('-vocabSize', 100000,'vocabulary size')
cmd:option('-sentenceLength', 5,'length of input sequences')
cmd:option('-learningRate', 0.01,'init learning rate')
cmd:option('-numEpochs', 5, 'number of epochs to train for')

local params = cmd:parse(arg)

if params.cuda == 1 then local useCuda = true else local useCuda = false end
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
local endIdx =  startIdx+minibatchSize-1 --(train.data:size()[1] - train.data:size()[1]) % minibatchSize
while(endIdx-1 < train.labels:size()[1])
do
    table.insert(dataBatches,toCuda(train.data:narrow(1, startIdx, endIdx-startIdx-1)))
    table.insert(labelsBatches,toCuda(train.labels:narrow(1, startIdx, endIdx-startIdx-1)))
    startIdx = endIdx
    endIdx = startIdx + minibatchSize+1
end

----- Set up network : We use a one-layer convnet and then max pooling across the time axis.
local net = nn.Sequential()
--- conv ---
--local convWidth = 3
--net:add(nn.LookupTable(vocabSize,embeddingDim))
--net:add(nn.TemporalConvolution(embeddingDim,embeddingDim,convWidth))
--net:add(nn.ReLU())
--net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis
--net:add(nn.Max(3))
--net:add(nn.Linear(embeddingDim, numClasses))
--net:add(nn.LogSoftMax())

---- polyglot sortof ----
--net:add(nn.LookupTable(vocabSize,embeddingDim))
--net:add(nn.Reshape(concatDim))
--net:add(nn.Tanh(concatDim))
--net:add(nn.Linear(concatDim, numClasses))
--net:add(nn.LogSoftMax())

---- nlp afs ----
net:add(nn.LookupTable(vocabSize,embeddingDim))
net:add(nn.Reshape(concatDim))
net:add(nn.Linear(concatDim, hiddenUnits))
net:add(nn.HardTanh())
net:add(nn.Linear(hiddenUnits, numClasses))
net:add(nn.LogSoftMax())


local criterion = nn.ClassNLLCriterion()
toCuda(criterion)
toCuda(net)


----Do the optimization. All relevant tensors should be on the GPU. (if using cuda)
local parameters, gradParameters = net:getParameters()
local startTime = sys.clock()

for epoch = 1, numEpochs
do
    print('Starting epoch ' .. epoch .. ' of ' .. numEpochs)
    -- TODO wtf is wrong with the end of this
    for i = 1, numBatches - 100
    do
        local idx = (i % numBatches) + 1
        local sentences = dataBatches[idx]
        local labels = labelsBatches[idx]
        local batch_error = 0
        local function fEval(x)
            if parameters ~= x then parameters:copy(x) end
            net:zeroGradParameters()
            local output = net:forward(sentences)
            local err = criterion:forward(output,labels)
            local df_do = criterion:backward(output, labels)
            net:backward(sentences, df_do)
            batch_error = batch_error+err
            return err, gradParameters
        end

        optimMethod(fEval, parameters, optConfig, optState)
        if(i % 50 == 0) then
            print(string.format('%f percent complete \t speed = %f examples/sec \t error = %f',
                i/(numEpochs * numBatches), (i*minibatchSize)/(sys.clock() - startTime), batch_error))
        end
    end
end

--- Evaluate ---
local test = torch.load(test_file)
local tp = 0
local fp = 0
local tn = 0
local fn = 0
for i = 1, test.labels:size()[1]
do
    local sample = toCuda(test.data:narrow(1, i, 1))
    local label = toCuda(test.labels:narrow(1, i, 1)[1])
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
