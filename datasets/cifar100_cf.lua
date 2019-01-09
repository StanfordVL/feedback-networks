--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

------------
-- This file is downloading and transforming CIFAR-100.
-- It is based on cifar10.lua
-- Ludovic Trottier
------------

local t = require 'datasets/transforms'

local M = {}
local CifarDataset = torch.class('resnet.CifarDataset', M)

function CifarDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
   self.coarse = opt.coarse and opt.testOnly
   self.newMapping = opt.newMapping
end

function CifarDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]
   local mapping = torch.Tensor({12, 0, 14, 5, 5, 8, 4, 4, 15, 19, 11, 14, 1, 16, 4, 2, 19, 1, 4, 2, 8, 2, 11, 9, 4, 8, 4, 5, 11, 5, 0, 2, 0, 6, 5, 14, 2, 16, 5, 18, 19, 10, 5, 2, 18, 4, 14, 6, 15, 9, 5, 3, 6, 12, 3, 5, 1, 12, 10, 6, 9, 11, 7, 5, 5, 5, 5, 0, 1, 17, 7, 9, 5, 0, 5, 3, 17, 4, 18, 4, 5, 16, 7, 12, 8, 10, 19, 8, 5, 15, 16, 0, 7, 0, 13, 0, 6, 5, 14, 18})
   mapping:add(1)
   local clabel = self.imageInfo.coarse[i]
   if self.newMapping then
      clabel = mapping[label]
   end

   return {
      input = image,
      target = label,
      ctarget = clabel,
   }
end

function CifarDataset:size()
   return self.imageInfo.data:size(1)
   -- return 512
end


-- Computed from entire CIFAR-100 training set with this code:
--      dataset = torch.load('cifar100.t7')
--      tt = dataset.train.data:double();
--      tt = tt:transpose(2,4);
--      tt = tt:reshape(50000*32*32, 3);
--      tt:mean(1)
--      tt:std(1)
local meanstd = {
   mean = {129.3, 124.1, 112.4},
   std  = {68.2,  65.4,  70.4},
}

function CifarDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CifarDataset
