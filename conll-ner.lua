----Toggle this flag if you want to use the GPU or not
local useCuda = false
----------------------

require 'torch'
require 'nn'
require 'optim'


if(useCuda) then
    require 'cunn'
    print('using GPU')
end
local function toCuda(x)
    local y = x
    if(useCuda) then
        y = x:cuda()
    end
    return y
end

local train_file = 'train'
local test_file = 'testa'

-----Define the problem size
-----We have a binary classification problem classifying 
-----random 'sentences' of length sentenceLength where there are vocabSzie possible words
local sentenceLength = 5
local vocabSize = 100000
local embeddingDim = 64
local convWidth = 5
local minibatchSize = 64
local numClasses = 8

---- Set up Data ----
local train = torch.load(train_file)
local numBatches = math.floor(train.data:size()[1]/minibatchSize)
print (numBatches)

-- split training data into batches
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
net:add(nn.LookupTable(vocabSize,embeddingDim))
net:add(nn.TemporalConvolution(embeddingDim,embeddingDim,convWidth))
net:add(nn.ReLU())
net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis
net:add(nn.Max(3))
net:add(nn.Linear(embeddingDim, numClasses))
net:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()
toCuda(criterion)
toCuda(net)


----Do the optimization. All relevant tensors should be on the GPU. (if using cuda)
local parameters, gradParameters = net:getParameters()
local optimMethod = optim.sgd
local startTime = sys.clock()

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

    optim.sgd(fEval, parameters)
    if(i % 15 == 0) then
        print(string.format('%f percent complete \t speed = %f examples/sec \t error = %f',
            i/numBatches, (i*minibatchSize)/(sys.clock() - startTime), batch_error))
    end
end


--- Evaluate ---
local test = torch.load(test_file)
local correct = 0
for i = 1, test.labels:size()[1]
do
    local sample = test.data:narrow(1, i, 1)
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
    print(label, max_index)
    if label == max_index then correct = correct + 1
    end
end
print(correct)
