########################################################################
#
# @author : Emmanouil Sylligardos & Paul Boniol
# @source : https://github.com/okrasolar/pytorch-timeseries/blob/master/
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/blocks
# @file : resnet_block
#
########################################################################

import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBlock
from models.layers.conv1d_same_padding import Conv1dSamePadding


class ResNetBlock(nn.Module):
	def __init__(
		self, 
		in_channels, 
		out_channels
	):
		super().__init__()

		channels = [in_channels, out_channels, out_channels, out_channels]
		kernel_sizes = [8, 5, 3]

		self.layers = nn.Sequential(*[
			ConvBlock(
				in_channels=channels[i], 
				out_channels=channels[i + 1],
				kernel_size=kernel_sizes[i], stride=1
			) for i in range(len(kernel_sizes))
		])

		self.match_channels = False
		if in_channels != out_channels:
			self.match_channels = True
			self.residual = nn.Sequential(*[
				Conv1dSamePadding(
					in_channels=in_channels, 
					out_channels=out_channels,
					kernel_size=1, 
					stride=1
				),
				nn.BatchNorm1d(num_features=out_channels)
			])

	def forward(self, x):
		if self.match_channels:
			return self.layers(x) + self.residual(x)
		return self.layers(x)