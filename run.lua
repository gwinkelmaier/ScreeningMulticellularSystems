----------------------------------------------------------------------
-- Main script for training a model for semantic segmentation
----------------------------------------------------------------------

require 'pl'
require 'nn'

----------------------------------------------------------------------
-- Local repo files
local opts = require 'opts'

-- Get the input arguments parsed and stored in opt
opt = opts.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')

require 'cudnn'
require 'cunn'
cutorch.setDevice(opt.devid)
print("Folder created at " .. opt.save)
os.execute('mkdir -p ' .. opt.save)

----------------------------------------------------------------------
print '==> load modules'
local data, chunks, ft
if opt.dataset == 'cv' then
   data  = require 'data/loadCamVid'
elseif opt.dataset == 'cs' then
   data = require 'data/loadCityscape'
elseif opt.dataset == 'su' then
   data = require 'data/loadSUN'
elseif opt.dataset == 'nc-r' then
   data = require 'loadNuclei-region'
elseif opt.dataset == 'nc-f' then
   data = require 'loadNuclei-fusion'
elseif opt.dataset == 'nc-p' then
   data = require 'loadNuclei-potential'
else
   error ("Dataset loader not found. (Available options are: cv/cs/su")
end

----------------------------------------------------------------------
print '==> training!'
local epoch = 1

t = paths.dofile(opt.model)

local train = require 'train'
local test  = require 'test'
while epoch < opt.maxepoch do
   local trainConf, model, loss = train(data.trainData, opt.dataClasses, epoch)
   test(data.testData, opt.dataClasses, epoch, trainConf, model, loss )
   trainConf = nil
   collectgarbage()
   epoch = epoch + 1
end
