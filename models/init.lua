--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'rnn'
require 'cunn'
require 'cudnn'
require 'ConvLSTM_bn'
require 'lib/ConvLSTM_bn2'
require 'lib/ConvLSTM_bn3'
require 'lib/ConvLSTM_bn3res'
require 'lib/ConvLSTM_bn3resSkip'
require 'lib/ConvLSTM_bn3resSkip2'
require 'lib/ConvLSTM_bn3resSkip3r'
require 'lib/ConvLSTM_bn3resSkip3'
require 'lib/ConvLSTM_bn3resSkip4'
require 'lib/ConvLSTM_bn3resSkipR'
require 'lib/ConvLSTM_bn4res'
require 'lib/ConvLSTM_bn_slim2'
require 'lib/ConvLSTM_bn_slim3'
require 'lib/ConvLSTM_resnet'
require 'lib/ConvLSTM_resnet2'
require 'lib/ConvLSTM_resnet3'
require 'lib/ConvLSTM_resnet4'
require 'lib/ConvLSTM_resnet5'
require 'lib/ConvLSTM_resnet5X3'
require 'lib/ConvLSTM_resnet5X4'
require 'lib/ConvLSTM_resnet52'
require 'lib/ConvLSTM_bottleneck5'
require 'lib/ConvLSTM_resnet5_new'
require 'lib/ConvLSTM_resnet5_new2'
require 'lib/ConvLSTM_resnet5a'
require 'lib/ConvLSTM_resnet5b'
require 'lib/ConvLSTM_resnet5c'
require 'lib/ConvLSTM_resnet5r'
require 'lib/ConvLSTM_resnet5ai'
require 'lib/ConvLSTM_resnet5noi'
require 'lib/ConvLSTM_resnet5nof'
require 'lib/ConvLSTM_resnet5noc'
require 'lib/ConvLSTM_resnet5noo'
require 'lib/ConvLSTM_resnet5aio'
require 'lib/ConvLSTM_resnet5aioc'
require 'lib/ConvLSTM_resnet5aiof'
require 'lib/ConvLSTM_resnet5aic'
require 'lib/ConvLSTM_resnet5aif'
require 'lib/ConvLSTM_resnet5aifc'
require 'lib/ConvLSTM_resnet5af'
require 'lib/ConvLSTM_resnet5afo'
require 'lib/ConvLSTM_resnet5afc'
require 'lib/ConvLSTM_resnet5afco'
require 'lib/ConvLSTM_resnet5ac'
require 'lib/ConvLSTM_resnet5aco'
require 'lib/ConvLSTM_resnet5L4'
require 'lib/ConvLSTM_resnet6'
require 'lib/ConvLSTM_resnet7'
require 'lib/ConvLSTM_resnet8'
require 'lib/ConvLSTM_resnet9'
require 'lib/ConvLSTM_resnet10'
require 'lib/ConvGRU'
require 'lib/ConvGRU2'
require 'lib/ConvGRU3res'
require 'lib/ConvRNN'
require 'lib/ConvRNN2'
require 'lib/ConvRNN3res'
require 'lib/ConvRNN_double'
require 'lib/shortCutLSTM'
require 'lib/CoarseFineCriterion'
require 'lib/CoarseFineCriterion2'
require 'lib/CoarseFineCriterion3'
require 'lib/CoarseFineCriterion4'
require 'lib/CoarseFineCriterion5'
require 'lib/CoarseFineCriterion6'
require 'lib/shortCutLSTM'
require 'lib/DecayFineCriterion'
require 'lib/DecayFineCriterion2'
require 'lib/DecayFineCriterion3'
require 'lib/LogNLLCriterion'

local M = {}

function M.setup(opt, checkpoint)
   local model, mmodel
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
--       if opt.testOnly then
--         model = require('models/' .. opt.netType)(opt):cuda()
--       else
      mmodel = torch.load(modelPath):cuda()
--       end
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      mmodel = torch.load(opt.retrain):cuda()
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      mmodel = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
     if opt.retrain ~='none' then 
       mmodel = mmodel:get(1)
     else    
       mmodel = model:get(1)
     end
   end

   --[[if opt.retrain ~='none' then
   print(opt.seqLength)
   -- modify the model and finetune
   model = nn.Sequential()
   for i = 1, 3 do
      layer = mmodel:get(i) 
      model:add(layer)
   end
      model:add(nn.Replicate(8))
      model:add(nn.Contiguous())
      model:add(nn.View(opt.seqLength, -1, 16, 32, 32))
      model:add(nn.SplitTable(1))           
    for i = 8, 15 do 
      layer = mmodel:get(i)
      model:add(layer)
    end
    end]]--
    model = mmodel:cuda()
   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local imsize = opt.dataset == 'imagenet' and 224 or 32
      local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      M.shareGradInput(model)
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:cuda())
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            require 'rnn'
            require 'ConvLSTM_bn'
            require 'lib/ConvLSTM_bn2'
            require 'lib/ConvLSTM_bn3'
            require 'lib/ConvLSTM_bn3res'
            require 'lib/ConvLSTM_bn3resSkip'
            require 'lib/ConvLSTM_bn3resSkip2'
            require 'lib/ConvLSTM_bn3resSkip3'
            require 'lib/ConvLSTM_bn3resSkip3r'
            require 'lib/ConvLSTM_bn3resSkip4'
            require 'lib/ConvLSTM_bn3resSkipR'
            require 'lib/ConvLSTM_bn4res'
            require 'lib/ConvLSTM_resnet'
            require 'lib/ConvLSTM_resnet2'
            require 'lib/ConvLSTM_resnet3'
            require 'lib/ConvLSTM_resnet4'
            require 'lib/ConvLSTM_resnet5'
            require 'lib/ConvLSTM_resnet5'
            require 'lib/ConvLSTM_resnet5X3'
            require 'lib/ConvLSTM_resnet5X4'
            require 'lib/ConvLSTM_bottleneck5'
            require 'lib/ConvLSTM_resnet5_new'
            require 'lib/ConvLSTM_resnet5_new2'
            require 'lib/ConvLSTM_resnet5a'
            require 'lib/ConvLSTM_resnet5b'
            require 'lib/ConvLSTM_resnet5c'
            require 'lib/ConvLSTM_resnet5r'
            require 'lib/ConvLSTM_resnet5ai'
            require 'lib/ConvLSTM_resnet5noi'
            require 'lib/ConvLSTM_resnet5nof'
            require 'lib/ConvLSTM_resnet5noc'
            require 'lib/ConvLSTM_resnet5noo'
            require 'lib/ConvLSTM_resnet5aio'
            require 'lib/ConvLSTM_resnet5aioc'
            require 'lib/ConvLSTM_resnet5aiof'
            require 'lib/ConvLSTM_resnet5aic'
            require 'lib/ConvLSTM_resnet5aif'
            require 'lib/ConvLSTM_resnet5aifc'
            require 'lib/ConvLSTM_resnet5af'
            require 'lib/ConvLSTM_resnet5afo'
            require 'lib/ConvLSTM_resnet5afc'
            require 'lib/ConvLSTM_resnet5afco'
            require 'lib/ConvLSTM_resnet5ac'
            require 'lib/ConvLSTM_resnet5aco'
            require 'lib/ConvLSTM_resnet5L4'
            require 'lib/ConvLSTM_resnet6'
            require 'lib/ConvLSTM_resnet7'
            require 'lib/ConvLSTM_resnet8'
            require 'lib/ConvLSTM_resnet9'
            require 'lib/ConvLSTM_resnet10'
            require 'lib/ConvRNN'
            require 'lib/ConvRNN2'
            require 'lib/ConvRNN3res'
            require 'lib/ConvGRU'
            require 'lib/ConvGRU2'
            require 'lib/ConvGRU3res'
            require 'lib/ConvRNN_double'
            require 'lib/shortCutLSTM'
            require 'lib/CoarseFineCriterion'
            require 'lib/CoarseFineCriterion2'
            require 'lib/CoarseFineCriterion3'
            require 'lib/CoarseFineCriterion4'
            require 'lib/CoarseFineCriterion5'
            require 'lib/CoarseFineCriterion6'
            require 'lib/shortCutLSTM'
            require 'lib/DecayFineCriterion'
            require 'lib/DecayFineCriterion2'
            require 'lib/DecayFineCriterion3'
            require 'lib/LogNLLCriterion'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   -- local criterion = nn.CrossEntropyCriterion():cuda()
   local criterion
   if opt.sequenceOut then
     criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion()):cuda()
   else
     criterion = nn.CrossEntropyCriterion():cuda()
   end
   return model, criterion
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

return M
