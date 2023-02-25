########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/layers
# @file : attention
#
########################################################################

import torch
import torch.nn as nn

class Attention(nn.Module):
	"""Multi-head Attention as described in the 'Attention is all you need' paper"""

	def __init__(self, dim, num_heads=8, qkv_bias=False, 
						attn_drop=0.0, proj_drop=0.0):
		super(Attention, self).__init__()
		
		assert(
			dim % num_heads == 0
		), "Embedding dimension should be divisible by number of heads"
		
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		# batch_size, n_tokens, dim
		B, N, C = x.shape

		# (3, batch_size, num_heads, n_tokens, dim // num_heads)
		qkv = (
			self.qkv(x)
			.reshape(B, N, 3, self.num_heads, C // self.num_heads)
			.permute(2, 0, 3, 1, 4)
		)
		q, k, v = qkv[0], qkv[1], qkv[2]
		
		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)

		return x

'''
# Test
attention = Attention(dim=256, num_heads=4)

x = torch.ones(128, 17, 256)
print(x.shape)
y = attention.forward(x)
print(y.shape)
'''