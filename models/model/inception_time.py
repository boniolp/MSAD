########################################################################
#
# @author : Emmanouil Sylligardos & Paul Boniol
# @source : https://github.com/TheMrGhostman/InceptionTime-Pytorch/blob/master/inception.py
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/model
# @file : inception_time
#
########################################################################

import torch
import torch.nn as nn

from typing import cast, Union, List

from models.blocks.inception_block import InceptionBlock


class InceptionModel(nn.Module):	
	def __init__(
		self, 
		num_blocks, 
		in_channels, 
		out_channels,
		bottleneck_channels, 
		kernel_sizes,
		use_residuals='default',
		num_pred_classes=1
	):
		super().__init__()
	
		self.input_args = {
			'num_blocks': num_blocks,
			'in_channels': in_channels,
			'out_channels': out_channels,
			'bottleneck_channels': bottleneck_channels,
			'kernel_sizes': kernel_sizes,
			'use_residuals': use_residuals,
			'num_pred_classes': num_pred_classes
		}

		if num_blocks == -1:
			if isinstance(kernel_sizes, list):
				num_blocks = len(kernel_sizes)
			elif isinstance(kernel_sizes, int):
				num_blocks = 1
			else:
				raise ValueError("Can't recognize 'kernel_sizes' type")

		if bottleneck_channels == -1:
			bottleneck_channels = out_channels // 2


		channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
																		  num_blocks))
		bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
																	 num_blocks))
		kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
		
		if use_residuals == 'default':
			use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
		use_residuals = cast(List[bool], self._expand_to_blocks(
			cast(Union[bool, List[bool]], use_residuals), num_blocks)
		)

		self.blocks = nn.Sequential(*[
			InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
						   residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
						   kernel_size=kernel_sizes[i]) for i in range(num_blocks)
		])

		self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

	@staticmethod
	def _expand_to_blocks(value,
						  num_blocks):
		if isinstance(value, list):
			assert len(value) == num_blocks
		else:
			value = [value] * num_blocks
		return value

	def forward(self, x):
		x = self.blocks(x).mean(dim=-1)
		return self.linear(x)