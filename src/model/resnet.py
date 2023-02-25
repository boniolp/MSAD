########################################################################
#
# @author : Emmanouil Sylligardos & Paul Boniol
# @source : https://github.com/okrasolar/pytorch-timeseries/blob/master/
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/model
# @file : resnet
#
########################################################################

import torch
import torch.nn as nn

from models.blocks.resnet_block import ResNetBlock


class ResNetBaseline(nn.Module):	
	def __init__(
		self, 
		in_channels=1, 
		mid_channels=64,
		num_pred_classes=12,
		num_layers=3,
	):
		super().__init__()

		self.input_args = {
			'in_channels': in_channels,
			'num_pred_classes': num_pred_classes
		}

		# Initiate layers list
		self.layers = []
		self.curr_out_channels = mid_channels

		# Input first layer
		self.layers.append(ResNetBlock(
			in_channels=in_channels, 
			out_channels=mid_channels
		))

		# Add more layers (the first one has already been added)
		for i in range(1, num_layers):
			if i % 3 == 2:
				self.layers.append(ResNetBlock(
					in_channels=self.curr_out_channels, 
					out_channels=self.curr_out_channels
				))
			else:
				self.layers.append(ResNetBlock(
					in_channels=self.curr_out_channels, 
					out_channels=self.curr_out_channels * 2
				))
				self.curr_out_channels *= 2

		# Integrate all layers into a Sequential arch
		self.layers = nn.Sequential(*self.layers)

		# Add final layer
		self.final = nn.Linear(self.curr_out_channels, num_pred_classes)
		

	def forward(self, x):
		x = self.layers(x)
		return self.final(x.mean(dim=-1))