########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/layers
# @file : prenorm
#
########################################################################

import torch
import torch.nn as nn


class PreNorm(nn.Module):
	"""Normalize tensor per sample and then apply fn on it"""
	def __init__(self, dim, fn):
		super(PreNorm, self).__init__()
		self.norm = nn.LayerNorm(dim)
		self.fn = fn


	def forward(self, x, **kwargs):
		# res = self.norm(x.float())
		# print(res)
		# res = self.fn(res)
		# return res
		return self.fn(self.norm(x), **kwargs)

''' 
# Test

def add4(x):
	return x + 4

x = torch.tensor([[[1, 2, 3], [0.3, 0.5, .7]], [[7, 10, 14], [540, 1000, 1500]]], dtype=torch.float64)
prenorm = PreNorm(3, add4)
print(x)
y = prenorm.forward(x)
print(y)
'''