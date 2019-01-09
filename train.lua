--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
require 'hdf5'
require 'image'
require 'itorch'
require 'nn'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   -- self.optimState = optimState or {
   --    learningRate = opt.LR,
   --    learningRateDecay = 0.0,
   --    momentum = opt.momentum,
   --    nesterov = true,
   --    dampening = 0.0,
   --    weightDecay = opt.weightDecay,
   -- }
   if opt.optim == 'rmsprop' then
      self.optimState = optimState or {
      learningRate = opt.LR,
      alpha = 0.8,
      epsilon = 1e-8
   }
   elseif opt.optim == 'adagrad' then
      self.optimState = optimState or {
      learningRate = opt.LR,
      epsilon = 1e-8
   }
   elseif opt.optim == 'sgd' then
      self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   elseif opt.optim == 'adam' then
      self.optimState = optimState or {
      learningRate = opt.LR,
      alpha = 0.8,
      beta = 0.999,
      epsilon = 1e-8
   }
   else
      error('Optim method is not implemented.')
   end

   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   print('total number of parameters : ', self.params:nElement())
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   self.model:forget()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- multiple output loss
      local target
      if self.opt.sequenceOut then
        local labels = self.target
        target = {}
        for i = 1, self.opt.seqLength do
          target[i] = labels
        end
      else
        target = self.target
      end
      local output, batchSize
      if self.opt.sequenceOut then
        output = self.model:forward(self.input)
        batchSize = output[1]:size(1)
      else
        output = self.model:forward(self.input):float()
        batchSize = output:size(1)
      end
      -- print(#output)
      local loss = self.criterion:forward(self.model.output, target)
      -- print(#self.model.output)
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, target)
      self.model:backward(self.input, self.criterion.gradInput)

      -- optim.sgd(feval, self.params, self.optimState)
      if self.opt.optim == 'rmsprop' then
        optim.rmsprop(feval, self.params, self.optimState)
      elseif self.opt.optim == 'adagrad' then
        optim.adagrad(feval, self.params, self.optimState)
      elseif self.opt.optim == 'sgd' then
        optim.sgd(feval, self.params, self.optimState)
      elseif self.opt.optim == 'adam' then
        optim.adam(feval, self.params, self.optimState)
      else
        error('Optim method is not implemented.')
      end
      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  LR %.0e  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, self.optimState.learningRate, loss, top1, top5))

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   -- local model_parameters, model_gradParameters = self.model:getParameters()
   -- print('total number of parameters : ', model_parameters:nElement())

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   predicted         = {}
   acc_total         = 0
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- multiple output loss
      local target
      if self.opt.sequenceOut then
        local labels = self.target
        target = {}
        for i = 1, self.opt.seqLength do
          target[i] = labels
        end
      else
        target = self.target
      end
      
      local output, batchSize
      if self.opt.sequenceOut then
        output = self.model:forward(self.input)
        batchSize = output[1]:size(1) / nCrops
      else
        output = self.model:forward(self.input):float()
        batchSize = output:size(1) / nCrops
      end

      local loss = self.criterion:forward(self.model.output, target)
      local top1, top5, pred = self:computeScore(output, sample.target, nCrops)

      -- for ii = 1, pred:size()[1] do
      --   pred_idx = N + ii
      --   print (pred_idx, pred[ii][1], sample.target[ii])
      --   if pred[ii][1] == sample.target[ii] then
      --     acc_total = acc_total + 1
      --   end
      --   predicted[tostring(pred_idx)] = torch.Tensor({pred[ii][1]})
      -- end

      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   -- print (acc_total)
   -- local predFile = hdf5.open(string.format('sm1_predicted_Iter%d.h5', self.opt.seqLength), 'w')
   -- predFile:write('features', predicted)
   -- predFile:close()

   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output_t, target, nCrops)
   local output
   if self.opt.sequenceOut then
     output = output_t[self.opt.seqLength]
   else
     output = output_t
   end

   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100, predictions
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      -- decay = epoch >= 62 and 2 or epoch >= 21 and 1 or 0
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      -- decay = epoch >= 192 and 2 or epoch >= 191 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
   -- return self.opt.LR * math.pow(0.5, decay)
end

return M.Trainer


