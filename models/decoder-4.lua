----------------------------------------------------------------------
-- Create model and calulate loss to optimize for decoder.
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
print '==> define parameters'

local histClasses = opt.datahistClasses
local classes = opt.dataClasses

----------------------------------------------------------------------
print '==> construct model'

nn.DataParallelTable.deserializeNGPUs = 1
model = torch.load(opt.CNNEncoder)
if torch.typename(model) == 'nn.DataParallelTable' then model = model:get(1) end
model:remove(#model.modules) -- remove the classifier

-- SpatialMaxUnpooling requires nn modules...
model:apply(function(module)
   if module.modules then
      for i,submodule in ipairs(module.modules) do
         if torch.typename(submodule):match('cudnn.VolumetricMaxPooling') then
            module.modules[i] = nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2) -- TODO: make more flexible
         end
      end
   end
end)

-- find pooling modules
local pooling_modules = {}
model:apply(function(module)
   if torch.typename(module):match('nn.VolumetricMaxPooling') then
      table.insert(pooling_modules, module)
   end
end)
assert(#pooling_modules == 2, 'There should be 2 pooling modules')

-- kill gradient
-- local grad_killer = nn.Identity()
-- function grad_killer:updateGradInput(input, gradOutput)
--    return self.gradInput:resizeAs(gradOutput):zero()
-- end
-- model:add(grad_killer)

-- decoder:

print(pooling_modules)

function bottleneck(input, output, upsample, reverse_module)
   local internal = output / 4
   local input_stride = upsample and 2 or 1

   local module = nn.Sequential()
   local sum = nn.ConcatTable()
   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(cudnn.VolumetricConvolution(input, internal, 1, 1, 1, 1, 1, 1, 0, 0, 0))
   main:add(nn.VolumetricBatchNormalization(internal, 1e-3))
   main:add(cudnn.ReLU(true))
   if not upsample then
      main:add(cudnn.VolumetricConvolution(internal, internal, 3, 3, 3, 1, 1, 1, 1, 1, 1))
   else
      main:add(nn.VolumetricFullConvolution(internal, internal, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1))
   end
   main:add(nn.VolumetricBatchNormalization(internal, 1e-3))
   main:add(cudnn.ReLU(true))
   main:add(cudnn.VolumetricConvolution(internal, output, 1, 1, 1, 1, 1, 1, 0, 0, 0))
   main:add(nn.VolumetricBatchNormalization(output, 1e-3))

   other:add(nn.Identity())
   if input ~= output or upsample then
      other:add(cudnn.VolumetricConvolution(input, output, 1, 1, 1, 1, 1, 1, 0, 0, 0))
      other:add(nn.VolumetricBatchNormalization(output, 1e-3))
      if upsample and reverse_module then
         other:add(nn.VolumetricMaxUnpooling(reverse_module))
      end
   end

   if upsample and not reverse_module then
      main:remove(#main.modules) -- remove BN
      return main
   end
   return module:add(sum):add(nn.CAddTable()):add(cudnn.ReLU(true))
end

--model:add(bottleneck(128, 128))
model:add(bottleneck(128, 64, true, pooling_modules[2]))         -- 32x64
--model:add(bottleneck(64, 64))
model:add(bottleneck(64, 64))

model:add(bottleneck(64, 16, true, pooling_modules[1]))          -- 64x128
--model:add(bottleneck(64, 16))
model:add(nn.VolumetricFullConvolution(16, #classes, 1,1,1,1,1,1))


if cutorch.getDeviceCount() > 1 then
   local gpu_list = {}
   for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end
   model = nn.DataParallelTable(1):add(model:cuda(), gpu_list)
   print(opt.nGPU .. " GPUs being used")
end


-- Loss: NLL
print('defining loss function:')
local normHist = histClasses / histClasses:sum()
local classWeights = torch.Tensor(#classes):fill(1)
for i = 1, #classes do
      classWeights[i] = 1 / (torch.log(1.04 + normHist[i]))
end

loss = cudnn.VolumetricCrossEntropyCriterion(classWeights)

model:cuda()
loss:cuda()

----------------------------------------------------------------------
print '==> here is the model:'
print(model)


-- return package:
return {
   model = model,
   loss = loss,
}

