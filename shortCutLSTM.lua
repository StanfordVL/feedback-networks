------------------------------------------------------------------------
--ConvGRU
------------------------------------------------------------------------
local _ = require 'moses'
require 'nn'
require 'dpnn'
require 'rnn'

local shortCutLSTM, parent = torch.class('nn.shortCutLSTM', 'nn.AbstractRecurrent')

function shortCutLSTM:__init(inputSize, outputSize, rho, Kr, stride)
   parent.__init(self, rho or 9999)
   self.inputSize = inputSize
   self.outputSize = outputSize   
   self.stride = stride or 1
   self.Kr = Kr
   -- build the model
   self.recurrentModule = self:buildModel()
   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule 
   
   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor() 
end

-------------------------- factory methods -----------------------------
function shortCutLSTM:buildModel()
   -- input : {input, prevOutput}
   -- output : {output}   
   if self.stride == 2 then
     model = nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, 2, 2))
            :add(nn.Concat(2)
                 :add(nn.Identity())
                 :add(nn.MulConstant(0)))
   else
     model = nn.Identity()
   end

   local shortCut = nn.Sequential()
   local para = nn.ParallelTable()
     -- para:add(model):add(nn.Identity())
     para:add(model):add(nn.Identity())
     shortCut:add(para)
     shortCut:add(nn.CAddTable())

   return shortCut
end

------------------------- forward backward -----------------------------
function shortCutLSTM:updateOutput(input)
   --print(#input)
   local prevOutput = self.zeroTensor
   if input:dim() == 4 then -- batch 
      self.zeroTensor:resize(input:size(1), self.outputSize, self.Kr, self.Kr):zero()
   else
      self.zeroTensor:resize(self.outputSize, self.Kr, self.Kr):zero()
   end
   --print(#prevOutput)
   -- output(t) = gru{input(t), output(t-1)}
   local output
   if self.train ~= false then
      -- self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output = recurrentModule:updateOutput{input, prevOutput}
   else
      output = self.recurrentModule:updateOutput{input, prevOutput}
   end
   
   self.outputs[self.step] = output
   
   self.output = output
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   -- note that we don't return the cell, just the output
   print (#self.outputs)
   return self.output
end

function shortCutLSTM:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)
   
   local gradInput
   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
   if self.gradPrevOutput then
      self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
      nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
      gradOutput = self._gradOutputs[step]
   end
   
   local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
   local inputTable = {input, output}
   local gradInputTable = recurrentModule:updateGradInput(inputTable, gradOutput)
   gradInput, self.gradPrevOutput = unpack(gradInputTable)
   if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
   
   return gradInput
end

function shortCutLSTM:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)
   
   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
   local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
   local inputTable = {input, output}
   local gradOutput = (step == self.step-1) and gradOutput or self._gradOutputs[step]
   recurrentModule:accGradParameters(inputTable, gradOutput, scale)
   return gradInput
end

function shortCutLSTM:__tostring__()
   return string.format('%s(s=%d t=%d)', torch.type(self), self.stride, self.rho)
end
