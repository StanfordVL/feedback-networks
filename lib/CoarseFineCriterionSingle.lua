require 'nn'
-- SUN Lin @Stanford 2016

local CoarseFineCriterionSingle, Criterion = torch.class('nn.CoarseFineCriterionSingle', 'nn.Criterion')

function CoarseFineCriterionSingle:__init(weights, fm, bm, num, den)
   Criterion.__init(self)
   self.lsm_f = nn.LogSoftMax()
   self.nll_f = nn.ClassNLLCriterion()
   
   self.lsm_c = nn.LogSoftMax()
   self.nll_c = nn.ClassNLLCriterion()
   
   self.fm = fm
   self.bm = bm
   self.num = num
   self.den = den
   self.cfratio = num / den
   print("Using nn.CoarseFineCriterionSingle")
end

function CoarseFineCriterionSingle:updateOutput(input, target_both)
   input = input:squeeze()
   local target = target_both.f
   target = type(target) == 'number' and target or target:squeeze()
   local ctarget = target_both.c
   ctarget = type(ctarget) == 'number' and ctarget or ctarget:squeeze()

   self.nll_f:updateOutput(input, target)
   temp = input:clone()

   cinput = (temp:exp() * self.fm):log()

   self.nll_c:updateOutput(cinput, ctarget)
   
   
   self.output = self.cfratio*self.nll_f.output + (1-self.cfratio)*self.nll_c.output
   return self.output
end

function CoarseFineCriterionSingle:updateGradInput(input, target_both)
   local size = input:size()
   local target = target_both.f
   target = type(target) == 'number' and target or target:squeeze()
   local ctarget = target_both.c
   ctarget = type(ctarget) == 'number' and ctarget or ctarget:squeeze()
   
   self.nll_f:updateGradInput(input, target)

   temp = input:clone()
   mapped = (temp:exp() * self.fm)
   cinput = mapped:log()
   loss= criterion_c:backward(cinput,ctarget)
   coarseGradInput = (loss:cmul(torch.ones(cinput:size()):cuda():cdiv(mapped)) * self.bm):cmul(temp):cuda()

   self.gradInput:view(self.cfratio*self.nll_f.gradInput + (1-self.cfratio)*coarseGradIput, size)
   return self.gradInput
end

return nn.CoarseFineCriterionSingle
