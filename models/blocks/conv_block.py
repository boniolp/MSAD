########################################################################
#
# @author : Emmanouil Sylligardos & Paul Boniol
# @source : https://github.com/okrasolar/pytorch-timeseries/blob/master/
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/blocks
# @file : conv_block
#
########################################################################


import torch
import torch.nn as nn

from models.layers.conv1d_same_padding import Conv1dSamePadding


class ConvBlock(nn.Module):
	def __init__(
		self, 
		in_channels, 
		out_channels, 
		kernel_size,
		stride
	):
		super().__init__()

		self.layers = nn.Sequential(
				Conv1dSamePadding(
					in_channels=in_channels,
					out_channels=out_channels,
					kernel_size=kernel_size,
					stride=stride),
				nn.BatchNorm1d(num_features=out_channels),
				nn.ReLU(),
		)

	def forward(self, x):
		return self.layers(x)