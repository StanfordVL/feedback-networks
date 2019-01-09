------------------------------------------------------------------------
--[[FineCoarseCriterion ]]--
-- Lin Sun 2016@Stanford 
------------------------------------------------------------------------
require 'nn'
require 'rnn'
local FineCoarseCriterion, parent = torch.class('nn.FineCoarseCriterion2_bc2', 'nn.Criterion')

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
   print('[INFO] using nn.FineCoarseCriterion2')
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

   for i=1,nStep do
      local criterion_f, criterion_c = self:getStepCriterion(i)
      ii = i
      num = ii
      -- ii  = i - 1
      -- num = ii * 2
      temp = input[i]:clone()
      
      cinput = nn.Log():forward(nn.Exp():forward(temp:float()) * self.fm:float())
      if true then
        self.output = self.output + (num/nStep)*0.5*criterion_f:forward(input[i], target[i]) + (1.0-num/nStep)*0.5*criterion_c:forward(cinput:cuda(), ctarget[i])
        -- self.output = self.output + criterion_f:forward(input[i], target[i]) + criterion_c:forward(cinput:cuda(), ctarget[i])
      else
        self.output = self.output + criterion_f:forward(input[i], target[i]) + 0*criterion_c:forward(cinput:cuda(), ctarget[i])
      end
      -- print("forward")
      -- print(criterion_f.output)
      -- print(criterion_c.output)
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
      ii = i
      num = ii
      -- ii = i - i
      -- num = ii * 2
      local criterion_f, criterion_c = self:getStepCriterion(i)
      fineGradInput[i] = criterion_f:backward(input[i], target[i])
      --print('input')
      --print(torch.sum(input[i]))
      temp = input[i]:clone()
      exp = nn.Exp()
      middle1 = exp:forward(temp:float())
      middle2 = middle1 * self.fm:float()
      cinput = nn.Log():forward(middle2)
      d_loss= criterion_c:backward(cinput:cuda(),ctarget[i]):float()
      --print('values')
      --print(torch.sum(loss))
      --print(torch.sum(value))
      --print(torch.sum(middle))
      d_cinput = nn.Log():backward(middle2, d_loss)
      d_middle1 = d_cinput * self.bm:float()
      d_temp = exp:backward(temp, d_middle1)
      coarseGradInput[i] = d_temp:cuda()
      -- print('max')
      print ('Iter', i)
      print('coarseGradInput_min:', torch.min(coarseGradInput[i]))
      print('fineGradInput_min:  ', torch.min(fineGradInput[i]))
      print('coarseGradInput_sum:', torch.sum(fineGradInput[i]))
      print('fineGradInput_sum:  ', torch.sum(coarseGradInput[i]))
      -- print (coarseGradInput[i])
      if torch.min(coarseGradInput[i]) > -0.005 then
        -- tableGradInput[i] = (num/nStep)*fineGradInput[i] + (1.0-num/nStep)*coarseGradInput[i]
        tableGradInput[i] = (num/nStep)*0.5*fineGradInput[i] + (1.0-num/nStep)*0.5*coarseGradInput[i]
      else
        -- tableGradInput[i] = 0*fineGradInput[i] + (1.0-num/nStep)*0.5*coarseGradInput[i]
        tableGradInput[i] = (num/nStep)*0.5*fineGradInput[i] + 0*coarseGradInput[i]
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
