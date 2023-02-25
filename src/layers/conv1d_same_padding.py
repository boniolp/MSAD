########################################################################
#
# @author : Emmanouil Sylligardos & Paul Boniol
# @source : https://github.com/okrasolar/pytorch-timeseries/blob/master/
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/layers
# @file : conv1d_same_padding
#
########################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F

class Conv1dSamePadding(nn.Conv1d):
	def conv1d_same_padding(
		self,
		inputs, 
		weight, 
		bias, 
		stride, 
		dilation, 
		groups
	):
		kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
		l_out = l_in = inputs.size(2)
		padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
		
		if padding % 2 != 0:
			inputs = F.pad(inputs, [0, 1])

		return F.conv1d(
			input=inputs, 
			weight=weight, 
			bias=bias, 
			stride=stride,
			padding=padding // 2,
			dilation=dilation, 
			groups=groups
		)

	def forward(self, inputs):
		return self.conv1d_same_padding(
			inputs, 
			self.weight, 
			self.bias, 
			self.stride,
			self.dilation, 
			self.groups
		)
