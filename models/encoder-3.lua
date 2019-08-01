----------------------------------------------------------------------
-- Create model and loss to optimize.
--
-- Mina Khoshdeli,
-- June 2017.
----------------------------------------------------------------------
matio = require 'matio'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'
require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------

----------------------------------------------------------------------
print '==> define parameters'

local histClasses = opt.datahistClasses
local classes = opt.dataClasses
local conClasses = opt.dataconClasses

----------------------------------------------------------------------
print '==> construct model'

local model = nn.Sequential()

local ct = 0
function _bottleneck(internal_scale, use_relu, asymetric, dilated, input, output, downsample)
   local internal = output / internal_scale
   local input_stride = downsample and 2 or 1

   local sum = nn.ConcatTable()

   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(cudnn.VolumetricConvolution(input, internal, input_stride,  input_stride, input_stride, input_stride, input_stride, input_stride, 0, 0, 0))
   main:add(nn.VolumetricBatchNormalization(internal))
   if use_relu then main:add(nn.PReLU(internal)) end
   if not asymetric and not dilated then
      main:add(cudnn.VolumetricConvolution(internal, internal, 3, 3, 3, 1, 1, 1, 1, 1, 1))
   elseif asymetric then
      local pad = (asymetric-1) / 2
      main:add(cudnn.VolumetricConvolution(internal, internal, asymetric, 1, 1, pad, 0, 0))
      main:add(cudnn.VolumetricConvolution(internal, internal, 1, asymetric,  1, 0, pad, 0))
      main:add(cudnn.VolumetricConvolution(internal, internal, 1, 1, asymetric,  0, 0, pad))
   elseif dilated then
      main:add(nn.VolumetricDilatedConvolution(internal, internal, 3, 3, 3, 1, 1, 1, dilated, dilated, dilated, dilated, dilated, dilated))
   else
      assert(false, 'You shouldn\'t be here')
   end
   main:add(nn.VolumetricBatchNormalization(internal))
   if use_relu then main:add(nn.PReLU(internal)) end
   main:add(cudnn.VolumetricConvolution(internal, output, 1, 1, 1, 1, 1, 1, 0, 0, 0))
   main:add(nn.VolumetricBatchNormalization(output, 1e-3))
   main:add(nn.VolumetricDropout((ct < 5) and 0.01 or 0.1))
   ct = ct + 1

   other:add(nn.Identity())
   if downsample then
      other:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
   end
   if input ~= output then
      other:add(nn.Padding(1, output-input, 3))
   end

   return nn.Sequential():add(sum):add(nn.CAddTable()):add(nn.PReLU(output))
end

local _ = require 'moses'
local bottleneck = _.bindn(_bottleneck, 4, true, false, false)
local cbottleneck = _.bindn(_bottleneck, 4, true, false, false)
local xbottleneck = _.bindn(_bottleneck, 4, true, 7, false)
local wbottleneck = _.bindn(_bottleneck, 4, true, 5, false)
local dbottleneck = _.bindn(_bottleneck, 4, true, false, 2)
local xdbottleneck = _.bindn(_bottleneck, 4, true, false, 4)
local xxdbottleneck = _.bindn(_bottleneck, 4, true, false, 8)
local xxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 16)
local xxxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 32)

local initial_block = nn.ConcatTable()
--Here is initial block with 16 feature maps and without downsampling
model:add(nn.VolumetricConvolution(1, 16, 3, 3, 3,1,1, 1,1,1,1))

--model:add(initial_block)                                         -- 128x256
--model:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
model:add(nn.VolumetricMaxPooling(2,2,2,2,2,2))
model:add(nn.VolumetricBatchNormalization(16))
model:add(nn.PReLU(16))
--model:add(bottleneck(16,16))
model:add(bottleneck(16, 64, true))                              -- 64x128
for i = 1,3 do
   model:add(bottleneck(64, 64))
end


model:add(bottleneck(64, 128, true))


--for i = 1,4 do
   model:add(cbottleneck(128, 128))
   model:add(dbottleneck(128, 128))
--   model:add(wbottleneck(128, 128))
--   model:add(xdbottleneck(128, 128))
--   model:add(cbottleneck(128, 128))
--   model:add(xxdbottleneck(128, 128))
--   model:add(wbottleneck(128, 128))
--   model:add(xxxdbottleneck(128, 128))
--end
model:add(cudnn.VolumetricConvolution(128, 2, 1, 1, 1))

model:cuda()

local gpu_list = {}
--for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end
--if opt.nGPU == 1 then
--   gpu_list[1] = opt.devid
--else
--   for i = 1, opt.nGPU do gpu_list[i] = i end
--end
--model = nn.DataParallelTable(1, true, true):add(model:cuda(), gpu_list)
--print(opt.nGPU .. " GPUs being used")

-- Loss: NLL
print('defining loss function:')
local normHist = histClasses / histClasses:sum()
local classWeights = torch.Tensor(#classes):fill(1)
for i = 1, #classes do
      classWeights[i] = 1 / (torch.log(1.2 + normHist[2-i+1]))
end


classWeights[1] = 0.05
classWeights[2] = 0.95

loss = cudnn.VolumetricCrossEntropyCriterion(classWeights)

loss:cuda()
----------------------------------------------------------------------
print '==> here is the model:'
print(model)


-- return package:
return {
   model = model,
   loss = loss,
}

