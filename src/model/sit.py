########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/model
# @file : sit
#
########################################################################

import torch
import torch.nn as nn

from models.embedding.patch_embed import EmbeddingStem
from models.blocks.transformer_encoder import TransformerEncoder
from models.layers.output_layer import OutputLayer


class SignalTransformer(nn.Module):
	def __init__(
		self,
		timeseries_size=512,
		window_size=16,
		in_channels=1,
		embedding_dim=256,
		num_layers=12,
		num_heads=8,
		qkv_bias=True,
		mlp_ratio=4.0,
		use_revised_ffn=False,
		dropout_rate=0.0,
		attn_dropout_rate=0.0,
		select_conv_linear=None,
		use_conv_patch=False,
		use_linear_patch=False,
		use_conv_stem=False,
		use_conv_stem_original=False,
		use_stem_scaled_relu=False,
		hidden_dims=None,
		cls_head=False,
		num_classes=12,
		representation_size=None,
	):

		super(SignalTransformer, self).__init__()

		if select_conv_linear is not None:
			if select_conv_linear:
				use_conv_patch = True
				use_linear_patch = False
			else:
				use_conv_patch = False
				use_linear_patch = True

		if use_stem_scaled_relu and not isinstance(hidden_dims, list):
			hidden_dims = [hidden_dims]

		# Embedding layer
		self.embedding_layer = EmbeddingStem(
			timeseries_size=timeseries_size,
			window_size=window_size,
			channels=in_channels,
			embedding_dim=embedding_dim,
			hidden_dims=hidden_dims,
			conv_patch=use_conv_patch,
			linear_patch=use_linear_patch,
			conv_stem=use_conv_stem,
			conv_stem_original=use_conv_stem_original,
			conv_stem_scaled_relu=use_stem_scaled_relu,
			position_embedding_dropout=dropout_rate,
			cls_head=cls_head,
		)

		# Transformer Encoder
		self.transformer = TransformerEncoder(
			dim=embedding_dim,
			depth=num_layers,
			heads=num_heads,
			mlp_ratio=mlp_ratio,
			attn_dropout=attn_dropout_rate,
			dropout=dropout_rate,
			qkv_bias=qkv_bias,
			revised=use_revised_ffn,
		)
		self.post_transformer_ln = nn.LayerNorm(embedding_dim)

		# Output layer
		self.cls_layer = OutputLayer(
			embedding_dim,
			num_classes=num_classes,
			representation_size=representation_size,
			cls_head=cls_head,
		)

	def forward(self, x):
		x = self.embedding_layer(x)
		x = self.transformer(x)
		x = self.post_transformer_ln(x)
		x = self.cls_layer(x)
		return x