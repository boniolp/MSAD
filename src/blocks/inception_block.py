########################################################################
#
# @author : Emmanouil Sylligardos & Paul Boniol
# @source : https://github.com/okrasolar/pytorch-timeseries/blob/master/
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/blocks
# @file : inception_block
#
########################################################################

import torch
import torch.nn as nn

from models.layers.conv1d_same_padding import Conv1dSamePadding


class InceptionBlock(nn.Module):
	def __init__(
		self, 
		in_channels, 
		out_channels,
		residual, 
		stride = 1, 
		bottleneck_channels = 32,
		kernel_size = 41
	):
		super().__init__()

		self.use_bottleneck = bottleneck_channels > 0
		if self.use_bottleneck:
			self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
												kernel_size=1, bias=False)
		kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
		start_channels = bottleneck_channels if self.use_bottleneck else in_channels
		channels = [start_channels] + [out_channels] * 3
		self.conv_layers = nn.Sequential(*[
			Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
							  kernel_size=kernel_size_s[i], stride=stride, bias=False)
			for i in range(len(kernel_size_s))
		])

		self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
		self.relu = nn.ReLU()

		self.use_residual = residual
		if residual:
			self.residual = nn.Sequential(*[
				Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
								  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm1d(out_channels),
				nn.ReLU()
			])

	def forward(self, x):
		org_x = x
		if self.use_bottleneck:
			x = self.bottleneck(x)
		x = self.conv_layers(x)

		if self.use_residual:
			x = x + self.residual(org_x)
		return x