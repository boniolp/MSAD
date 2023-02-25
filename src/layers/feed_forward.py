########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/layers
# @file : feed_forward
#
########################################################################

import torch
import torch.nn as nn


class FeedForward(nn.Module):
	"""
	Implementation of MLP for transformer
	"""

	def __init__(self, dim, hidden_dim, dropout_rate=0.0, revised=False):
		super(FeedForward, self).__init__()
		if not revised:
			"""
			Original: https://arxiv.org/pdf/2010.11929.pdf
			"""
			self.net = nn.Sequential(
				nn.Linear(dim, hidden_dim),
				nn.GELU(),
				nn.Dropout(p=dropout_rate),
				nn.Linear(hidden_dim, dim),
			)
		else:
			"""
			Scaled ReLU: https://arxiv.org/pdf/2109.03810.pdf
			"""
			self.net = nn.Sequential(
				nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1),
				nn.BatchNorm1d(hidden_dim),
				nn.GELU(),
				nn.Dropout(p=dropout_rate),
				nn.Conv1d(hidden_dim, dim, kernel_size=1, stride=1),
				nn.BatchNorm1d(dim),
				nn.GELU(),
			)

		self.revised = revised
		self._init_weights()

	def _init_weights(self):
		for name, module in self.net.named_children():
			if isinstance(module, nn.Linear):
				nn.init.normal_(module.bias, std=1e-6)

	def forward(self, x):
		if self.revised:
			x = x.permute(0, 2, 1)
			x = self.net(x)
			x = x.permute(0, 2, 1)
		else:
			x = self.net(x)

		return x

'''
# (batch_size, n_tokens, dim)
x = torch.ones(128, 17, 256)

ffn = FeedForward(dim=256, 
				hidden_dim=128, 
				dropout_rate=0.0, 
				revised=False)
print(x.shape)
y = ffn.forward(x)
print(y.shape)
'''