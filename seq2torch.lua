--[[
This takes a file where each line is a sentence. Example lines below. 

The first int is a class label for the sentence. 
Then, there is a tab. 
Then there is a space-separated list of word indices for the sentence. 

***This script expects that each sentence is of the same length. 
To massage your data into this form, you can take two approaches:
1) In a preprocessing script, pad your sentences with dummy tokens so that they're all the same length. 
2) Move your corpus around so that there is one file per possible sentence length, and call this conversion script on each file. Then, when training your downstream
model, load from one of these files at a time, so that the examples in your minibatch are all of the same size. 

71      1 117 2497 789 4 2
80      1 9 2838 2087 91 2
43      1 26 12208 8043 120 2
52      1 783 12208 4 5 2yst
48      1 2454 84 27 1162 2
49      1 4 8 24 131 2
71      1 9 992 148 4 2

--]]

require 'torch'
cmd = torch.CmdLine()
cmd:option('-file','','input file')
cmd:option('-len','','uniform length for each sequence')
cmd:option('-out','','out')


local params = cmd:parse(arg)
local expectedLen = params.len
local fname = params.file
local outFile = params.out
local numLines = tonumber(io.popen(string.format('wc -l %s | cut -d" " -f1',fname)):read("*a"))

--os.execute(string.format('wc -l %s | cut -d" " -f1',fname))
print(string.format('num input lines = %d',numLines))

local labels = torch.Tensor(numLines)
local data = torch.Tensor(numLines,expectedLen)

local lineIdx = 0
for line in io.lines(fname) do
	lineIdx = lineIdx + 1
	local ctr = 0
	for wordIdx in string.gmatch(line,'(%d+)%s*') do
		local idx = tonumber(wordIdx)
		if(ctr == 0) then
			labels[lineIdx] = idx
		else
			--print(string.format("idx = %d got %d, expected %d",wordIdx,ctr,expectedLen))
			data[lineIdx][ctr] = idx
		end
		ctr = ctr + 1
	end
	assert(ctr == expectedLen+1,string.format("got %d, expected %d",ctr,expectedLen))
end

local stuff = {
	labels = labels,
	data = data

}

torch.save(outFile,stuff)

