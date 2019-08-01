----------------------------------------------------------------------
-- data loader,
-- Mina Khoshdeli,
-- June 2017
----------------------------------------------------------------------
matio = require 'matio'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local trainFile = opt.datapath .. '/train.txt'
local testFile = opt.datapath .. '/test.txt'

----------------------------------------------------------------------

local classes = { 'Nuclei', 'Background'}

local conClasses = {'Nuclei', 'Background'}

print('==> number of classes: ' .. #classes ..', classes: ', classes)

----------------------------------------------------------------------
-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):fill(0)

local trainData
local testData

-- Function to read txt file and return image and ground truth path
function getPath(filepath)
   print("Extracting file names from: " .. filepath)
   local file = io.open(filepath, 'r')
   local imgPath = {}
   local gtPath = {}
   file:read()    -- throw away first line
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      col1 = opt.datapath .. col1
      col2 = opt.datapath .. col2
      table.insert(imgPath, col1)
      table.insert(gtPath, col2)
      fline = file:read()
   end
   return imgPath, gtPath
end

----------------------------------------------------------------------
-- Main section
local loadedFromCache = false
local cacheDir = paths.concat(opt.cachepath, 'Nuclei')
local NucleiCachePath = paths.concat(opt.cachepath, 'Nuclei', 'data.t7')
--local camvidCachePath = '/home/mkhoshdeli/SRC/Torch/ENet-training-master/train/data/media/camVid/data.t7'
if not paths.dirp(cacheDir) then paths.mkdir(cacheDir) end

if opt.cachepath ~= "none" and paths.filep(NucleiCachePath) then
   print('Loading cache data')
   local dataCache = torch.load(NucleiCachePath)
   assert(dataCache.trainData ~= nil, 'No trainData')
   assert(dataCache.testData ~= nil, 'No testData')
   trainData = dataCache.trainData
   testData = dataCache.testData
   histClasses = dataCache.histClasses
   loadedFromCache = true
   dataCache = nil
   collectgarbage()
else
   ----------------------------------------------------------------------
   -- Acquire image and ground truth paths for training set
   local imgPath, gtPath = getPath(trainFile)

   -- initialize data structures:
   trainData = {
      data = torch.FloatTensor(#imgPath, opt.channels, opt.imHeight , opt.imWidth, opt.imBand),
      labels = torch.FloatTensor(#imgPath, opt.labelHeight , opt.labelWidth, opt.labelBand),
      preverror = 1e10, -- a really huge number
      size = function() return trainData.data:size(1) end
   }

   print "==> Loading traning data"
   for i = 1, #imgPath do
      -- load original image
      local Img = matio.load(imgPath[i])
      local rawImg = torch.Tensor(opt.channels,opt.imHeight, opt.imWidth, opt.imBand)
            rawImg[{{1,opt.channels},{},{},{}}] = Img.patch

   
         trainData.data[i] = rawImg[{{1,opt.channels},{},{},{}}]

      -- load corresponding ground truth
      Img = matio.load(gtPath[i])
      rawImg = Img.m:squeeze():float()
     -- local mask = rawImg:eq(3):float()
     -- rawImg = rawImg - mask * 2

      if (opt.labelHeight == rawImg:size(1)) and
         (opt.labelWidth == rawImg:size(2))  and
	(opt.labelBand == rawImg:size(3)) then
         trainData.labels[i] = rawImg
      else
	local im = torch.Tensor(opt.labelWidth, opt.labelHeight,opt.imBand)
	local ims = torch.Tensor(opt.labelWidth, opt.labelHeight, opt.labelBand)
	for j = 1, opt.imBand do
		im[{{},{},j}] = image.scale(rawImg[{{},{},j}],opt.labelWidth, opt.labelHeight, 'simple')
	end
	for l = 1, opt.labelWidth do
		ims[{l,{},{}}] = image.scale(im[{l,{},{}}], opt.labelHeight, opt.labelBand, 'simple')
	end
         trainData.labels[i] = ims
      end
      histClasses = histClasses + torch.histc(trainData.labels[i], #classes, 1, #classes)
      xlua.progress(i, #imgPath)
      collectgarbage()
   end

   ----------------------------------------------------------------------
   -- Acquire image and ground truth paths for testing set
   local imgPath, gtPath = getPath(testFile)

   -- initialize data structures:
   testData = {
      data = torch.FloatTensor(#imgPath, opt.channels, opt.imHeight , opt.imWidth, opt.imBand),
      labels = torch.FloatTensor(#imgPath, opt.labelHeight , opt.labelWidth, opt.labelBand),
      preverror = 1e10, -- a really huge number
      size = function() return testData.data:size(1) end
   }

   print "==> Loading traning data"
   for i = 1, #imgPath do
      -- load original image
      local Img = matio.load(imgPath[i])
      local rawImg = torch.Tensor(opt.channels,opt.imHeight, opt.imWidth, opt.imBand)
            rawImg[{{1,opt.channels},{},{},{}}] = Img.patch

         testData.data[i] = rawImg[{{1,opt.channels},{},{},{}}]

      -- load corresponding ground truth
      Img = matio.load(gtPath[i])
      rawImg = Img.m:squeeze():float()
     -- local mask = rawImg:eq(3):float()
     -- rawImg = rawImg - mask * 2

      if (opt.labelHeight == rawImg:size(1)) and
         (opt.labelWidth == rawImg:size(2))  and
        (opt.labelBand == rawImg:size(3)) then
         testData.labels[i] = rawImg
      else
        local im = torch.Tensor(opt.labelWidth, opt.labelHeight,opt.imBand)
        local ims = torch.Tensor(opt.labelWidth, opt.labelHeight, opt.labelBand)
        for j = 1, opt.imBand do
                im[{{},{},j}] = image.scale(rawImg[{{},{},j}],opt.labelWidth, opt.labelHeight, 'simple')
        end
        for l = 1, opt.labelWidth do
                ims[{l,{},{}}] = image.scale(im[{l,{},{}}], opt.labelHeight, opt.labelBand, 'simple')
        end
         testData.labels[i] = ims
      end
      histClasses = histClasses + torch.histc(testData.labels[i], #classes, 1, #classes)
      xlua.progress(i, #imgPath)
      collectgarbage()
   end
collectgarbage()
end

----------------------------------------------------------------------
if opt.cachepath ~= "none" and not loadedFromCache then
    print('==> saving data to cache: ' .. NucleiCachePath)
    local dataCache = {
       trainData = trainData,
       testData = testData,
       histClasses = histClasses,
    }
    torch.save('data.t7', dataCache)
    dataCache = nil
    collectgarbage()
end

----------------------------------------------------------------------

----------------------------------------------------------------------

local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
   table.insert(classes_td, cat .. ',1\n')
end

--local file = io.open(paths.concat('/save/trained/model/', 'categories.txt'), 'w')
--file:write(table.concat(classes_td))
--file:close()

-- Exports
opt.dataClasses = classes
opt.dataconClasses = conClasses
opt.datahistClasses = histClasses

return {
   trainData = trainData,
   testData = testData
}
