--------------------------------------------------------------------
---- Mina Khoshdeli,
---- March 2018.
-------------------------------------------------------------------
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'image'
require 'torch'
matio = require 'matio'


fileDir = '/home/gwinkelmaier/parvin-labs/nuc-seg/'

p= 1
nn.DataParallelTable.deserializeNGPUs = 1

model = torch.load('trained-models/model-fusion.net')

channels = 3

local function Rescale(img)
m= img:min()
M = img:max()

if (m==M) then
        s = img:size()
        img = torch.zeros(s[1],s[2])
else
        img = (img-m)/(M-m)
end

return img
end

letters = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N'}

-- Evaluate Un-augmented Training Images
for t = 1 , 496, 1 do
input1 = image.load(fileDir .. 'data/Mina/inputs/trainSampleRGB' .. t ..'.png')
x = torch.Tensor(1, channels, 224, 224)
x[{1,{},{},{}}] = input1[{{1,channels},{1,224},{1,224}}]

model:evaluate()
x = x:cuda()
y = model:forward(x)
out1 = torch.DoubleTensor(2,224,224):copy(torch.squeeze(y[{1,{},{},{}}]))
matio.save(fileDir .. 'data/Mina/inputs/train-output-fusion' .. t .. '.mat', out1)
end

-- Evaluate augmented Training Images
for t = 1 , 496, 1 do
   for j=1,14,1 do 
   input1 = image.load(fileDir .. 'data/Mina/inputs/trainSampleRGB' .. t .. '_' .. letters[j] .. '.png')
   x = torch.Tensor(1, channels, 224, 224)
   x[{1,{},{},{}}] = input1[{{1,channels},{1,224},{1,224}}]
   
   model:evaluate()
   x = x:cuda()
   y = model:forward(x)
   out1 = torch.DoubleTensor(2,224,224):copy(torch.squeeze(y[{1,{},{},{}}]))
   matio.save(fileDir .. 'data/Mina/inputs/train-output-fusion' .. t .. '_' .. letters[j] .. '.mat', out1)
   end
end


-- Evaluate Un-augmented Testing Images
for t = 1 , 16, 1 do
input1 = image.load(fileDir .. 'data/Mina/inputs/testSampleRGB' .. t ..'.png')
x = torch.Tensor(1, channels, 224, 224)
x[{1,{},{},{}}] = input1[{{1,channels},{1,224},{1,224}}]

model:evaluate()
x = x:cuda()
y = model:forward(x)
out1 = torch.DoubleTensor(2,224,224):copy(torch.squeeze(y[{1,{},{},{}}]))
matio.save(fileDir .. 'data/Mina/inputs/test-output-fusion' .. t .. '.mat', out1)
end

-- Evaluate augmented Testing Images
for t = 1 , 16, 1 do
   for j=1,14,1 do 
   input1 = image.load(fileDir .. 'data/Mina/inputs/testSampleRGB' .. t .. '_' .. letters[j] .. '.png')
   x = torch.Tensor(1, channels, 224, 224)
   x[{1,{},{},{}}] = input1[{{1,channels},{1,224},{1,224}}]
   
   model:evaluate()
   x = x:cuda()
   y = model:forward(x)
   out1 = torch.DoubleTensor(2,224,224):copy(torch.squeeze(y[{1,{},{},{}}]))
   matio.save(fileDir .. 'data/Mina/inputs/test-output-fusion' .. t .. '_' .. letters[j] .. '.mat', out1)
   end
end
