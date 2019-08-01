----------------------------------------------------------------------
-- Create model and loss to optimize.
--
-- Adam Paszke,
-- May 2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '==> define parameters'

local histClasses = opt.datahistClasses
local classes = opt.dataClasses
local conClasses = opt.dataconClasses

----------------------------------------------------------------------
print '==> construct model'

local model = nn.Sequential()

local ct = 0
model:add(cudnn.SpatialConvolution(3,32,3,3,1,1,1,1))
model:add(nn.PReLU(32))
model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(32,64,3,3,1,1,1,1))
model:add(nn.PReLU(64))
model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))
model:add(nn.PReLU(128))
model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(128, 2, 1, 1))

local gpu_list = {}
--for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end
if opt.nGPU == 1 then
   gpu_list[1] = opt.devid
else
   for i = 1, opt.nGPU do gpu_list[i] = i end
end
model = nn.DataParallelTable(1, true, true):add(model:cuda(), gpu_list)
print(opt.nGPU .. " GPUs being used")

-- Loss: NLL
print('defining loss function:')
local normHist = histClasses / histClasses:sum()
local classWeights = torch.Tensor(#classes):fill(1)
for i = 1, #classes do
      classWeights[i] = 1 / (torch.log(1.2 + normHist[i]))
end

loss = cudnn.SpatialCrossEntropyCriterion(classWeights)

--loss = nn.AbsCriterion()

--loss = nn.MSECriterion()

loss:cuda()
----------------------------------------------------------------------
print '==> here is the model:'
print(model)


-- return package:
return {
   model = model,
   loss = loss,
}

