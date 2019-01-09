------------------------------------------------------------------------
--[[DecayFineCriterion]]--
-- Lin Sun 2016@Stanford 
------------------------------------------------------------------------
require 'nn'
require 'rnn'
local DecayFineCriterion, parent = torch.class('nn.DecayFineCriterion', 'nn.Criterion')

function DecayFineCriterion:__init(criterion, fm, bm, nStep)
   parent.__init(self)
   self.criterion_f = criterion
   self.criterion_c = criterion
   self.fm = fm
   self.bm = bm 
   print(#fm, #bm)
  
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("SequencerCriterion shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a SequencerCriterion. "..
         "Its modules can also be similarly decorated with a Sequencer.")
   end
   self.clones_f = {}
   self.clones_c = {}
   self.gradInput = {}
   print('[INFO] using nn.DecayFineCriterion')
   print ('[INFO] F-C Ratio')
   local preSum = 0.0
   for i=1,nStep do 
     ii = (nStep-(i-1))
     preSum = preSum + (1.0+(1.0/ii))
   end
   self.preSum = preSum
   for i=1,nStep do
      -- print (1/i, '*Fine + ', 0, '* Coarse')
      ii = (nStep-(i-1))
      -- print (1/ii, '*Fine + ', 0, '* Coarse')
      term = 1.0+(1.0/ii)
      term = term / preSum * nStep
      print (term, '*Fine + ', 0, '* Coarse')
   end
end

function DecayFineCriterion:getStepCriterion(step)
   assert(step, "expecting step at arg 1")
   local criterion_f = self.clones_f[step]
   local criterion_c = self.clones_c[step]
   if not criterion_f or not criterion_c then
      criterion_f = self.criterion_f:clone()
      self.clones_f[step] = criterion_f
      
      criterion_c = self.criterion_c:clone()
      self.clones_c[step] = criterion_c
   end
   return criterion_f, criterion_c
end

function DecayFineCriterion:updateOutput(input, target_both)
   self.output = 0
   local target = target_both.f
   local ctarget = target_both.c
   local nStep
   if torch.isTensor(input) then
      assert(torch.isTensor(target), "expecting target Tensor since input is a Tensor")
      assert(target:size(1) == input:size(1), "target should have as many elements as input")
      nStep = input:size(1)
   else
      assert(torch.type(target) == 'table', "expecting target table")
      assert(#target == #input, "target should have as many elements as input")
      nStep = #input
   end

   --print(#input, #input[1], #target[1]) 
   for i=1,nStep do
      local criterion_f, criterion_c = self:getStepCriterion(i)
      ii = (nStep-(i-1))
      -- self.output = self.output + (1/i)*criterion_f:forward(input[i], target[i]) + 0*criterion_c:forward(input[i]*self.fm, ctarget[i])
      -- self.output = self.output + (1/ii)*criterion_f:forward(input[i], target[i]) + 0*criterion_c:forward(input[i]*self.fm, ctarget[i])
      term = 1.0+(1.0/ii)
      term = term / self.preSum * nStep
      self.output = self.output + (term)*criterion_f:forward(input[i], target[i]) + 0*criterion_c:forward(input[i]*self.fm, ctarget[i])
   end
   
   return self.output
end

function DecayFineCriterion:updateGradInput(input, target_both)
   local target = target_both.f
   local ctarget = target_both.c
   self.gradInput = {}
   if torch.isTensor(input) then
      assert(torch.isTensor(target), "expecting target Tensor since input is a Tensor")
      assert(target:size(1) == input:size(1), "target should have as many elements as input")
      nStep = input:size(1)
   else
      assert(torch.type(target) == 'table', "expecting gradOutput table")
      assert(#target == #input, "target should have as many elements as input")
      nStep = #input
   end
   
   local fineGradInput = {}
   local coarseGradInput = {}
   local tableGradInput = {}
   for i=1,nStep do
      ii = (nStep-(i-1))
      local criterion_f, criterion_c = self:getStepCriterion(i)
      fineGradInput[i] = criterion_f:backward(input[i], target[i])
      coarseGradInput[i] = criterion_c:backward(input[i]*self.fm, ctarget[i])*self.bm
      -- tableGradInput[i] = (1/i)*fineGradInput[i] + 0*coarseGradInput[i]
      -- tableGradInput[i] = (1/ii)*fineGradInput[i] + 0*coarseGradInput[i]
      term = 1.0+(1.0/ii)
      term = term / self.preSum * nStep
      tableGradInput[i] = (term)*fineGradInput[i] + 0*coarseGradInput[i]
   end
   
   if torch.isTensor(input) then
      self.gradInput = tableGradInput[1].new()
      self.gradInput:resize(nStep, unpack(tableGradInput[1]:size():totable()))
      for step=1,nStep do
         self.gradInput[step]:copy(tableGradInput[step])
      end
   else
      self.gradInput = tableGradInput
   end
   
   return self.gradInput
end
