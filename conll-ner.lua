require 'torch'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-cuda', 0,'whether to use gpu')
-- data file locations
cmd:option('-train', 'data/conll2003/eng.train.torch','torch format train file list')
cmd:option('-test', 'data/conll2003/eng.testa.torch','torch format test file list')
cmd:option('-loadEmbeddings', '', 'file containing serialized torch embeddings')
cmd:option('-saveModel', '', 'file to save the trained model to')
cmd:option('-labelMap', 'data/conll2003/label-map.index','file containing map from label strings to index. needed for entity level evaluation')
-- model / data sizes
cmd:option('-labelDim', 8,'label dimension')
cmd:option('-embeddingDim', 64,'embedding dimension')
cmd:option('-hiddenDim', 300,'hidden layer dimension')
cmd:option('-vocabSize', 100004,'vocabulary size')
cmd:option('-sentenceLength', 5,'length of input sequences')
cmd:option('-batchSize', 64,'minibatch size')
-- optimization
cmd:option('-learningRate', 0.01, 'init learning rate')
cmd:option('-tanh', false, 'use tanh layer, hardTanh otherwise')
cmd:option('-adagrad', false, 'use adagrad to optimize, sgd otherwise')
cmd:option('-hinge', false, 'use hinge loss while training, nll otherwise')
cmd:option('-stopEarly', false, 'stop training early if evaluation F1 goes down')
cmd:option('-numEpochs', 5, 'number of epochs to train for')
cmd:option('-evaluateFrequency', 5, 'number of epochs to train for')

local params = cmd:parse(arg)

useCuda = params.cuda == 1
if(useCuda) then
    require 'cunn'
    print('using GPU')
else
    require 'nn'
    print ('using CPU')
end

local function toCuda(x) return useCuda and x:cuda() or x end

--- data parameters
local train_file = params.train
local test_file = params.test
local sentenceLength = params.sentenceLength
local vocabSize = params.vocabSize
local train = torch.load(train_file)

--- model parameters
local embeddingDim = params.embeddingDim
local hiddenUnits = params.hiddenDim
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
local optimMethod = params.adagrad and optim.adagrad or optim.sgd
local numEpochs = params.numEpochs
local numBatches = math.floor(train.data:size(1)/params.batchSize)

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
if params.tanh then net:add(nn.Tanh()) else net:add(nn.HardTanh()) end
net:add(nn.Linear(hiddenUnits, numClasses))
net:add(nn.LogSoftMax())

local criterion = params.hinge and nn.MultiMarginCriterion() or nn.ClassNLLCriterion()
toCuda(criterion)
toCuda(net)

--- Evaluate ---
local function predict(sample)
    local pred = net:forward(sample)
    -- find the label with the max score
    local max_prob = -math.huge
    local max_index = -math.huge
    for c = 1, numClasses
    do
        if pred[c] > max_prob then
            max_prob = pred[c]
            max_index = c
        end
    end
    return max_index
end

--- evaluate scores by chunking entities with BIO tags
local function evaluate()
    local label_index_str_map = {}
    local label_str_index_map = {}
    for line in io.lines(params.labelMap) do
        local label_string, label_index = string.match(line, "([^\t]+)\t([^\t]+)")
        label_index_str_map[tonumber(label_index)] = label_string
        label_str_index_map[label_string] = tonumber(label_index)
    end
    local O_index = label_str_index_map["O"]

    print ('Evaluating')
    local test = torch.load(test_file)
    local tp = 0
    local fp = 0
    local tn = 0
    local fn = 0
    local i = 1
    while (i <test.labels:size()[1])
    do
        local s = toCuda(test.data:select(1, i))
        local l = test.labels:select(1, i)
        i = i + 1
        -- O, just score
        if (l == O_index) then
            if predict(s) == l then tn = tn + 1 else fp = fp + 1 end
        else
        -- score the entire entity
            local correct = true
            if predict(s) ~= l then correct = false end
            local last_l = l
            -- replace B-* tags with I-* since they are only important for delimeting
            if (string.sub(label_index_str_map[l], 1, 1) == 'B') then
                local last_l = label_str_index_map[label_index_str_map[l]:gsub("^%l", 'I')]
            end
            s = toCuda(test.data:select(1, i))
            l = test.labels:select(1, i)
            -- while this token is still part of the current entity
            while (last_l == l) do
                if predict(s) ~= l then correct = false end
                i = i + 1
                s = toCuda(test.data:select(1, i))
                l = test.labels:select(1, i)
            end
            if correct then tp = tp + 1 else fn = fn + 1 end
        end
    end

    local precision = tp / (tp + fp)
    local recall = tp / (tp + fn)
    local f1 = 2 * ((precision * recall) / (precision + recall))
    print(string.format('F1 : %f\t Recall : %f\tPrecision : %f', f1, recall, precision))
    return f1
end

local function evaluate_per_token()
    print ('Evaluating per token accuracy')
    local test = torch.load(test_file)
    local tp = 0
    local fp = 0
    local tn = 0
    local fn = 0
    for i = 1, test.labels:size()[1]
    do
        -- get a score for each label on this sample
        local sample = toCuda(test.data:select(1, i))
        local label = test.labels:select(1, i)
        local max_index = predict(sample)
        -- update counts
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
    local data_batches = {}
    local label_batches = {}
    local start = 1
    while start <= train.labels:size(1) do
        local size = math.min(params.batchSize, train.labels:size(1) - start + 1)
        table.insert(label_batches, toCuda(train.labels:narrow(1, start, size)))
        table.insert(data_batches, toCuda(train.data:narrow(1, start, size)))
        start = start + size
    end

    local parameters, gradParameters = net:getParameters()
    local last_f1 = 0.0
    for epoch = 1, numEpochs
    do
        -- randomly shuffle mini batches
        local shuffle = torch.randperm(numBatches)
        local epoch_error = 0
        local startTime = sys.clock()
        io.write('Starting epoch ', epoch, ' of ', numEpochs, '\n')
        for i = 1, numBatches
        do
            local idx = shuffle[i]
            local sentences = data_batches[idx]
            local labels = label_batches[idx]

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

            if(i % 50 == 0) then
                io.write(string.format('\r%.3f percent complete\tspeed = %.2f examples/sec',
                    i/(numBatches), (i*params.batchSize)/(sys.clock() - startTime)))
                io.flush()
            end
        end
        print(string.format('\nEpoch error = %f', epoch_error))
        if (epoch % params.evaluateFrequency == 0 or epoch == params.numEpochs) then
            local f1 = evaluate()
            -- end training early if f1 goes down
            if params.stopEarly and f1 < last_f1 then break else last_f1 = f1 end
            -- save the trained model if location specified
            if params.saveModel ~= '' then torch.save(params.saveModel, net) end
        end
    end
    if params.saveModel ~= '' then torch.save(params.saveModel, net) end
end

train_model()