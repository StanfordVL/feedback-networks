------------------------------------------------------------------------
--[[DecayFineCriterion]]--
-- Lin Sun 2016@Stanford 
------------------------------------------------------------------------
require 'nn'
require 'rnn'
local DecayFineCriterion, parent = torch.class('nn.DecayFineCriterion2', 'nn.Criterion')

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
   print('[INFO] using nn.DecayFineCriterion2')
   print ('[INFO] F-C Ratio')
   for i=1,nStep do
   --    print (1/torch.sqrt(i), '*Fine + ', 0, '* Coarse')
     if i < 4 then  
       print (1.0, '*Fine + ', 0, '* Coarse')
     elseif i == 4 then
       print (0.9, '*Fine + ', 0, '* Coarse')
     else
       print (0.8, '*Fine + ', 0, '* Coarse')
     end
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
      if i < 4 then
        self.output = self.output + (1.0)*criterion_f:forward(input[i], target[i]) + 0*criterion_c:forward(input[i]*self.fm, ctarget[i])
      elseif i == 4 then
        self.output = self.output + (0.9)*criterion_f:forward(input[i], target[i]) + 0*criterion_c:forward(input[i]*self.fm, ctarget[i])
      else
        self.output = self.output + (0.8)*criterion_f:forward(input[i], target[i]) + 0*criterion_c:forward(input[i]*self.fm, ctarget[i])
      end
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
      local criterion_f, criterion_c = self:getStepCriterion(i)
      fineGradInput[i] = criterion_f:backward(input[i], target[i])
      coarseGradInput[i] = criterion_c:backward(input[i]*self.fm, ctarget[i])*self.bm
      if i < 4 then
        tableGradInput[i] = (1.0)*fineGradInput[i] + 0*coarseGradInput[i]
      elseif i == 4 then
        tableGradInput[i] = (0.9)*fineGradInput[i] + 0*coarseGradInput[i]
      else
        tableGradInput[i] = (0.8)*fineGradInput[i] + 0*coarseGradInput[i]
      end
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
