########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/layers
# @file : output_layer
#
########################################################################

import torch
import torch.nn as nn


class OutputLayer(nn.Module):
	def __init__(self, 
				embedding_dim, 
				num_classes=12, 
				representation_size=None,
				cls_head=False):
		super(OutputLayer, self).__init__()

		modules = []
		if representation_size:
			modules.append(nn.Linear(embedding_dim, representation_size))
			modules.append(nn.Tanh())
			modules.append(nn.Linear(representation_size, num_classes))
		else:
			modules.append(nn.Linear(embedding_dim, num_classes))

		self.net = nn.Sequential(*modules)

		if cls_head:
			self.to_cls_token = nn.Identity()

		self.cls_head = cls_head
		self.num_classes = num_classes
		self._init_weights()

	def _init_weights(self):
		for name, module in self.net.named_children():
			if isinstance(module, nn.Linear):
				if module.weight.shape[0] == self.num_classes:
					nn.init.zeros_(module.weight)
					nn.init.zeros_(module.bias)

	def forward(self, x):
		if self.cls_head:
			# Feed to the output layer only the cls_token
			# (batch_size, n_tokens, dim) -> (batch_size, dim)
			x = self.to_cls_token(x[:, 0])
		else:
			"""
			Scaling Vision Transformer: https://arxiv.org/abs/2106.04560
			"""
			# Feed to the output layer the mean of all tokens
			x = torch.mean(x, dim=1)

		return self.net(x)

'''
# (batch_size, n_tokens, dim)
x = torch.ones(128, 17, 256)

output = Outputlayer(256, 12, cls_head=True)

print(x.shape)
y = output.forward(x)
print(y.shape)
'''