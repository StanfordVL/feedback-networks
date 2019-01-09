local LogNLLCriterion, parent = torch.class("nn.LogNLLCriterion", "nn.Criterion")

function LogNLLCriterion:__init()
   self.inputModule =  nn.Log()
   if self.inputModule then
      local params = self.inputModule:parameters()
      if params and #params > 0 then
         print"Warning: nn.LogNLLCriterion doesn't support parameter updates"
      end
   end
   self.criterion = nn.ClassNLLCriterion()
end

function LogNLLCriterion:updateOutput(input, target)
   --print(input)
   if self.inputModule then
      self.input = self.inputModule:forward(input)
   end
   --print(self.input)
   self.output = self.criterion:forward(self.input or input, target)
   return self.output
end

function LogNLLCriterion:updateGradInput(input, target)
   self.gradInput = self.criterion:backward(self.input or input,  target)
   if self.inputModule then
      self.gradInput = self.inputModule:backward(input, self.gradInput)
   end
   return self.gradInput
end

function LogNLLCriterion:type(type, typecache)
   if self.inputModule then
      self.inputModule:type(type, typecache)
   end
   self.criterion:type(type, typecache)
   return parent.type(self, type, typecache)
end
