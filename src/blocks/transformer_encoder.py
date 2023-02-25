########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/blocks
# @file : transformer_encoder
#
########################################################################

import torch
import torch.nn as nn

from models.layers.attention import Attention
from models.layers.feed_forward import FeedForward
from models.layers.prenorm import PreNorm


class TransformerEncoder(nn.Module):
	def __init__(
		self,
		dim,
		depth,
		heads,
		mlp_ratio=4.0,
		attn_dropout=0.0,
		dropout=0.0,
		qkv_bias=True,
		revised=False,
	):
		super().__init__()
		self.layers = nn.ModuleList([])

		assert isinstance(
			mlp_ratio, float
		), "MLP ratio should be a float for valid "
		mlp_dim = int(mlp_ratio * dim)

		for _ in range(depth):
			self.layers.append(
				nn.ModuleList(
					[
						PreNorm(
							dim,
							Attention(
								dim,
								num_heads=heads,
								qkv_bias=qkv_bias,
								attn_drop=attn_dropout,
								proj_drop=dropout,
							),
						),
						PreNorm(
							dim,
							FeedForward(
								dim,
								mlp_dim,
								dropout_rate=dropout,
								revised=False
							),
						)
						if not revised
						else FeedForward(
								dim,
								mlp_dim,
								dropout_rate=dropout,
								revised=True
							)
					]
				)
			)
	
	def forward(self, x):
		for attn, ff in self.layers:
			x = attn(x) + x
			x = ff(x) + x
		return x