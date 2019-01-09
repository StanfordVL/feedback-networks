------------------------------------------------------------------------
--[[FineCoarseCriterion]]--
-- Lin Sun 2016@Stanford 
------------------------------------------------------------------------
require 'nn'
require 'rnn'
local FineCoarseCriterion, parent = torch.class('nn.FineCoarseCriterion', 'nn.Criterion')

function FineCoarseCriterion:__init(criterion, fm, bm)
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
   print('[INFO] using nn.FineCoarseCriterion')
   print ('[INFO] F-C Ratio')
   nStep = 4
   for i=1,nStep do
      ii = i - 1
      nnStep = nStep - 1
      print (ii/nnStep, '*Fine + ', 1-ii/nnStep, '* Coarse')
   end
end

function FineCoarseCriterion:getStepCriterion(step)
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

function FineCoarseCriterion:updateOutput(input, target_both)
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
      ii = i - 1
      nnStep = nStep - 1
      local criterion_f, criterion_c = self:getStepCriterion(i)
      temp = input[i]:clone()
      cinput = (temp:exp() * self.fm):log()
      self.output = self.output + (ii/nnStep)*criterion_f:forward(input[i], target[i]) + (1-ii/nnStep)*criterion_c:forward(cinput, ctarget[i])
   end
   
   return self.output
end

function FineCoarseCriterion:updateGradInput(input, target_both)
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
      ii = i - 1
      nnStep = nStep - 1
      local criterion_f, criterion_c = self:getStepCriterion(i)
      fineGradInput[i] = criterion_f:backward(input[i], target[i])
      temp = input[i]:clone()
      cinput = (temp:exp() * self.fm):log()
      loss= criterion_c:backward(cinput,ctarget[i])
      coarseGradInput[i] = (loss:cmul(torch.ones(cinput:size()):cuda():cdiv(temp * self.fm)) * self.bm):cmul(temp):cuda()
      tableGradInput[i] = (ii/nnStep)*fineGradInput[i] + (1 - ii/nnStep)*coarseGradInput[i]
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
