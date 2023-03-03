########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/embedding
# @file : patch_embed
#
########################################################################

import torch
import torch.nn as nn

import math
import warnings


class EmbeddingStem(nn.Module):
	def __init__(
		self,
		timeseries_size=512,
		window_size=16,
		channels=1,
		embedding_dim=256,
		hidden_dims=None,
		conv_patch=False,
		linear_patch=False,
		conv_stem=True,
		conv_stem_original=True,
		conv_stem_scaled_relu=False,
		position_embedding_dropout=0,
		cls_head=True,
	):
		super(EmbeddingStem, self).__init__()

		assert(
			sum([conv_patch, conv_stem, linear_patch]) == 1
		), "Only one of three models should be active"

		assert(
			timeseries_size % window_size == 0
		), 'Timeseries size should be divisible by the window size'

		assert not(
			conv_stem and cls_head
		), "Cannot use [CLS] token approach with full conv stems for ViT"

		# Prepare step
		if linear_patch or conv_patch:
			num_windows = timeseries_size // window_size
			self.window_size = window_size

			if cls_head: 
				self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
				num_windows += 1

			# Positional embedding
			self.pos_embed = nn.Parameter(
				torch.zeros(1, num_windows, embedding_dim)
			)
			self.pos_drop = nn.Dropout(p=position_embedding_dropout)

		# The 3 modes of embedding
		if conv_patch:
			self.projection = nn.Sequential(
				nn.Conv1d(
					in_channels=channels,
					out_channels=embedding_dim,
					kernel_size=window_size,
					stride=window_size,
				),
			)
		elif linear_patch:
			patch_dim = channels * window_size
			self.projection = nn.Sequential(
				# Rearrange(
				# 	'b c (l w) -> b l (w c)',
				# 	c=channels,
				# 	w=window_size,
				# ),
				nn.Linear(patch_dim, embedding_dim)
			)
		elif conv_stem:
			assert(
				conv_stem_scaled_relu ^ conv_stem_original
			), "Can use either the original or the scaled relu stem"

			if not isinstance(hidden_dims, list):
				raise ValueError("Cannot create stem without list of sizes")

			if conv_stem_original:
				"""
				Conv stem from https://arxiv.org/pdf/2106.14881.pdf
				"""
				hidden_dims.insert(0, channels)
				modules = []
				
				for i, (in_ch, out_ch) in enumerate(
					zip(hidden_dims[:-1], hidden_dims[1:])
				):
					# print(f"{i}: {in_ch} --> {out_ch}")
					modules.append(
						nn.Conv1d(
							in_ch,
							out_ch,
							kernel_size=3,
							stride=2 if in_ch != out_ch else 1,
							padding=1,
							bias=False,
						),
					)
					modules.append(nn.BatchNorm1d(out_ch),)
					modules.append(nn.ReLU(inplace=True))

				modules.append(
					nn.Conv1d(
						hidden_dims[-1], embedding_dim, kernel_size=1, stride=1,
					),
				)
				self.projection = nn.Sequential(*modules)
			elif conv_stem_scaled_relu:
				"""
				Conv stem from https://arxiv.org/pdf/2109.03810.pdf
				"""
				assert(
					len(hidden_dims) == 1
				), "Only one value for hidden_dim is allowed"
				mid_ch = hidden_dims[0]

				# fmt: off
				self.projection = nn.Sequential(
					nn.Conv1d(
						channels,
						mid_ch,
						kernel_size=7,
						stride=2,
						padding=3,
						bias=False,
					),
					nn.BatchNorm1d(mid_ch),
					nn.ReLU(inplace=True),
					nn.Conv1d(
						mid_ch,
						mid_ch,
						kernel_size=3,
						stride=1,
						padding=1,
						bias=False,
					),
					nn.BatchNorm1d(mid_ch),
					nn.ReLU(inplace=True),
					nn.Conv1d(
						mid_ch,
						mid_ch,
						kernel_size=3,
						stride=1,
						padding=1,
						bias=False,
					),
					nn.BatchNorm1d(mid_ch),
					nn.ReLU(inplace=True),
					nn.Conv1d(
						mid_ch,
						embedding_dim,
						kernel_size=window_size // 2,
						stride=window_size // 2,
					),
				)
				#fmt: on

			else:
				raise ValueError("Undefined convolutional stem type")

		self.conv_stem = conv_stem
		self.conv_patch = conv_patch
		self.linear_patch = linear_patch
		self.cls_head = cls_head

		self._init_weights()

	def _init_weights(self):
		if not self.conv_stem:
			trunc_normal_(self.pos_embed, std=0.02)

	def forward(self, x):
		if self.conv_stem:
			x = self.projection(x)
			x = x.flatten(2).transpose(1, 2)
			return x
		elif self.linear_patch:
			B, C, L = x.size()
			x = x.reshape(B, L // self.window_size, C*self.window_size)
			x = self.projection(x)
		elif self.conv_patch:
			x = self.projection(x)
			x = x.flatten(2).transpose(1, 2)
		if self.cls_head:
			cls_token = self.cls_token.expand(x.shape[0], -1, -1)
			x = torch.cat((cls_token, x), dim=1)

		return self.pos_drop(x + self.pos_embed)


def _no_grad_trunc_normal(tensor, mean, std, a, b):

	def norm_cdf(x):
		return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

	if (mean < a - 2 * std) or (mean > b - 2 * std):
		warings.warn(
			"mean is more than 2 std form [a ,b] in nn.init.trunc_normal_."
			"The distribution of values may be incorrect.",
			stacklevel=2
		)

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values form [l, u], then
		# translate to [2l-1, 2u-1]
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal dist.
		# to get truncated standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.0))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)

		return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
	# type: (Tensor, float, float, float, float) -> Tensor
	r"""Fills the input Tensor with values drawn from a truncated
	normal distribution. The values are effectively drawn form the
	normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
	with values outside :math:`[a, b]` redrawn until they are within
	the bounds. The method used for generating the random values works
	best when :math:`a \leq \text{mean} \leq b`.
	Args:
		tensor: an n-dimensional `torch.Tensor`
		mean: the mean of the normal distribution
		std: the standard deviation of the normal distribution
		a: the minimum cutoff value
		b: the maximum cutoff value
	Examples:
		>>> w = torch.empty(3, 5)
		>>> nn.init.trunc_normal_(w)
	"""
	return _no_grad_trunc_normal(tensor, mean, std, a, b)