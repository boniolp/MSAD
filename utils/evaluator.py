########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : evaluator
#
########################################################################

from utils.timeseries_dataset import TimeseriesDataset
from utils.config import *

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from collections import Counter
from time import perf_counter
from tqdm import tqdm


class Evaluator:
	"""A class with evaluation tools
	"""

	def predict(
		self,
		model,
		fnames,
		data_path,
		batch_size
	):
		"""Predict function for all the deep learning models
		(ConvNet, ResNet, InceptionTime, SignalTransformer).

		:param model: the object model whose predictions we want
		:param fnames: the names of the timeseries to be predicted
		:param data_path: the path to the timeseries 
			(please check that path and fnames together make the complete path)
		:param batch_size: the batch size used to make the predictions
		:return df: dataframe with timeseries and predictions per time series
		"""

		# Setup
		all_preds = []

		loop = tqdm(
			fnames, 
			total=len(fnames),
			desc=f"Computing({batch_size})",
			unit="files",
			leave=True
		)

		# Main loop
		for fname in loop:
			# Fetch data for this specific timeserie
			data = TimeseriesDataset(
				data_path=data_path,
				fnames=[fname],
				verbose=False
			)
			preds = self.predict_timeseries(model, data, batch_size=batch_size, device='cuda')
			
			# Compute metric value
			counter = Counter(preds)
			most_voted = counter.most_common(1)
			
			# Save info
			all_preds.append(detector_names[most_voted[0][0]])
		
		fnames = [x[:-4] for x in fnames]

		return pd.DataFrame(data=all_preds, columns=["class"], index=fnames)


	def predict_timeseries(self, model, val_data, batch_size, device='cuda', k=1):
		all_preds = []
		
		# Timeserie to batches
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

		for (inputs, labels) in val_loader:
			# Move data to the same device as model
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			# Make predictions
			outputs = model(inputs.float())

			# Compute topk acc
			preds = outputs.argmax(dim=1)
			all_preds.extend(preds.tolist())

		return all_preds

'''
	def predict_non_deep(self, model, X_val, y_val):
		all_preds = []
		all_acc = []
		
		# Make predictions
		preds = model.predict(X_val)

		# preds = outputs.argmax(dim=1)
		# acc = (y_val == preds).sum() / y_val.shape[0]

		# all_acc.append(acc)
		all_preds.extend(preds.tolist())

		return all_preds
'''