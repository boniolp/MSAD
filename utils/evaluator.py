########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: utils
# @file : evaluate_experiment.py
#
########################################################################

from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.scores_loader import ScoresLoader
from utils.timeseries_dataset import TimeseriesDataset
from utils.config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from time import perf_counter
from tqdm import tqdm


class Evaluator:
	"""A class with evaluation tools
	"""

	def __init__(self):
		class_weight_true = torch.Tensor([6.7638, 19.1447,  0.6702,  2.2743,  0.9210,  0.7753,  9.0709,  1.3917, 0.2167,  2.8573,  0.8394,  1.9684]).to('cuda')
		self.criterion_true = nn.CrossEntropyLoss(
			weight=class_weight_true
		).to('cuda')

		class_weight_false = torch.Tensor([ 0.8136,  1.5609,  1.4061, 11.6383,  6.0874,  0.1789,  0.4932,  3.6057, 1.5700,  9.7459,  4.1053,  3.4235]).to('cuda')
		self.criterion_false = nn.CrossEntropyLoss(
			weight=class_weight_false
		).to('cuda')

	def predict(
		self,
		model,
		fnames,
		data_path,
		batch_size
	):
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
			preds = self.predict_timeserie(model, data, batch_size=batch_size, device='cuda')
			
			# Compute metric value
			counter = Counter(preds)
			most_voted = counter.most_common(1)
			
			# Save info
			all_preds.append(detector_names[most_voted[0][0]])
		
		fnames = [x[:-4] for x in fnames]

		return pd.DataFrame(data=all_preds, columns=["class"], index=fnames)


	def compute_anom_score_simple(
		self,
		model,
		model_type,
		fnames,
		metric_values,
		metric,
		data_path,
		batch_size
	):
		assert(
			len(fnames) == len(metric_values)
		), "Sizes not the same {}, {}".format(len(fnames), len(metric_values))

		# Setup
		pred_scores = []
		inf_time = []
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

			# Predict timeseries
			if model_type == 'deep':
				tic = perf_counter()
				preds = self.predict_timeserie(model, data, self.criterion_true, batch_size=batch_size, device='cuda')
			else:
				X_val, y_val = data.__getallsamples__().astype('float32'), data.__getalllabels__()
				tic = perf_counter()
				preds = self.predict_non_deep(model, X_val, y_val)
			
			# Compute metric value
			counter = Counter(preds)
			most_voted = counter.most_common(1)
			toc = perf_counter()

			# Save info
			inf_time.append(toc-tic)
			all_preds.append(detector_names[most_voted[0][0]])
			pred_scores.append(metric_values.loc[fname[:-4]].iloc[most_voted[0][0]])

		assert(
			len(fnames) == len(pred_scores)
		), "Sizes not the same {}, {}".format(len(fnames), len(pred_scores))
		fnames = [x[:-4] for x in fnames]

		return pd.DataFrame(data=zip(pred_scores, all_preds, inf_time), columns=["score", "class", "inf"], index=fnames)

	def compute_anomaly_score_non_deep(
		self,
		model,
		fnames,
		labels,
		scores,
		metric,
		data_path,
		window_size,
		keep_labels=True
	):
		all_acc = []
		all_model_scores = []
		model_metrics = []
		scores_loader = ScoresLoader('DUMMY_PATH')
		k = 1

		assert(
			len(fnames) == len(labels) == len(scores)
		), "Sizes not the same {}, {}, {}".format(len(fnames), len(labels), len(scores))

		loop = tqdm(
			zip(fnames, labels, scores), 
			total=len(fnames),
			desc="Computing",
			unit="files",
			leave=True
		)
		for fname, label, score in loop:
			# Fetch data for this speecific timeserie
			data = TimeseriesDataset(
				data_path=data_path,
				fnames=[fname],
				keep_labels=keep_labels,
				verbose=False
			)

			X_val, y_val = data.__getallsamples__().astype('float32'), data.__getalllabels__()

			# Create predictions
			acc, preds = self.predict_non_deep(model, X_val, y_val)
			all_acc.append(acc)
			loop.set_postfix(val_acc=np.mean(all_acc))
			preds = [[int(x)] for x in preds]

			# Compute score from predictions
			if keep_labels:
				model_scores = self.compute_score_genie(preds, score, k=1, strategy='majority')
			else:
				model_scores = self.compute_score_overlord(preds, score, window_size, k=1)
				label = np.concatenate(self.split_ts(label, window_size))
			all_model_scores.append(model_scores)

			# Compute metric
			curr_model_metrics = scores_loader.compute_single_sample(label, np.vstack(model_scores.values()).T, metric)
			model_metrics.append({i:curr_model_metrics[i] for i in range(0, k)})	

		# Transform to dataframe
		fnames = [x[:-4] for x in fnames]
		df = pd.DataFrame.from_records(model_metrics, index=fnames)

		return df, all_model_scores, all_acc


	def compute_anomaly_score(
		self, 
		model,
		fnames,
		labels,
		scores,
		metric,
		keep_labels,
		data_path,
		window_size,
	):
		scores_loader = ScoresLoader('DUMMY_PATH')
		all_acc = []
		all_loss = []
		all_distance = []
		all_model_scores = []
		model_metrics = []
		k = 1

		assert(
			len(fnames) == len(labels) == len(scores)
		)

		loop = tqdm(
			zip(fnames, labels, scores), 
			total=len(fnames),
			desc="Computing",
			unit="files",
			leave=True
		)
		for fname, label, score in loop:
			# Fetch data for this speecific timeserie
			data = TimeseriesDataset(
				data_path=data_path,
				fnames=[fname],
				keep_labels=keep_labels,
				verbose=False
			)

			# Create predictions
			if keep_labels:
				acc, loss, distance, preds = self.predict_timeserie(model, data, self.criterion_true, batch_size=32, device='cuda', k=12)
			else:
				acc, loss, distance, preds = self.predict_timeserie(model, data, self.criterion_false, batch_size=32, device='cuda', k=12)
			all_acc.append(acc)
			all_loss.append(loss)
			all_distance.append(distance)
			loop.set_postfix(
				val_acc=np.mean(all_acc),
				val_loss=np.mean(all_loss),
				val_distance=np.mean(all_distance),
			)

			if keep_labels:
				model_scores = self.compute_score_genie(preds, score, k=k, strategy='majority')
			else:
				model_scores = self.compute_score_overlord(preds, score, window_size, k=k)
				label = np.concatenate(self.split_ts(label, window_size))
		
			# Compute metric
			curr_model_metrics = scores_loader.compute_single_sample(label, np.vstack(model_scores.values()).T, metric)
			model_metrics.append({i:curr_model_metrics[i] for i in range(0, k)})	

		# Transform to dataframe
		fnames = [x[:-4] for x in fnames]
		df = pd.DataFrame.from_records(model_metrics, index=fnames)

		return df, all_acc, all_loss, all_distance


	def compute_score_genie(self, preds, score, k=4, strategy='majority'):
		'''Compute anomaly score for Genie models'''
		anomaly_score = {i:[] for i in range(0, k)}
		
		if strategy == 'majority':
			curr_preds = [x[0] for x in preds]
			# Count top k voted
			counter = Counter(curr_preds)

			for i in range(k):
				# Pick k most voted detectors
				most_voted_detectors = [x[0] for x in counter.most_common()[:i+1]]

				# Average their scores
				anomaly_score[i] = np.mean(score[:, most_voted_detectors], axis=1)

		elif strategy == 'rank':
			tmp_ranking = {i:0 for i in range(0, 12)}

			for pred in preds:
				for p in pred:
					tmp_ranking[p] += pred.index(p)

			tmp_ranking = {k: v for k, v in sorted(tmp_ranking.items(), key=lambda item: item[1], reverse=False)}
			for i in range(k):
				highest_ranked_detectors = [x for x in list(tmp_ranking.keys())[:i+1]]
				
				# Average their scores
				anomaly_score[i] = np.mean(score[:, highest_ranked_detectors], axis=1)
		else:
			raise ValueError('Not such strategy {}'.format(strategy))

		return anomaly_score



	def compute_score_overlord(self, preds, score, window_size=None, k=4):
		'''Compute anomaly score for Overlord models'''
		anomaly_score = {k:[] for k in range(0, k)}

		# Infer window size if not given
		if window_size is None:
			bin_powers = [2**i for i in range(0, 15)]
			window_size = math.floor(score.shape[0] / len(preds))
			window_size = min(bin_powers, key=lambda x:abs(x-window_size))
		
		# Split scores
		score_split = self.split_ts(score, window_size)
		assert(
			len(preds) == score_split.shape[0]
		), "Predictions and score splits lengths should match"

		for i in range(k):
			curr_preds = [x[:i+1] for x in preds]

			# Compute average of topk detectors per window
			for window_score, window_pred in zip(score_split, curr_preds):
				anomaly_score[i] += list(np.mean(window_score[:, window_pred], axis=1))
			anomaly_score[i] = np.array(anomaly_score[i])

		return anomaly_score


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



	def predict_timeserie(self, model, val_data, batch_size, device='cuda', k=1):
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


	def split_timeserie(self, data, window_size):
		'''Splits a timeserie in windows of window size. 
		If the window size does not perfectly divide the length
		of the time serie the modulo is thrown away from 
		its beginning.
		'''

		# Compute the modulo
		modulo = data.shape[0] % window_size

		# Throw away the rest
		rest = data[:modulo]
		data = data[modulo:]
		
		# Compute the number of windows
		k = data.shape[0] / window_size
		assert(math.ceil(k) == k)

		# Split everything into windows
		data_split = np.asarray(np.split(data, k))
		
		return data_split, rest


	def split_ts(self, data, window_size):
		'''Split a timeserie into windows according to window_size.
		If the timeserie can not be divided exactly by the window_size
		then the first window will overlap the second.

		:parapm data: the timeserie to be segmented
		:parapm window_size: the size of the windows
		'''
		# Compute the modulo
		modulo = data.shape[0] % window_size

		# Compute the number of windows
		k = data[modulo:].shape[0] / window_size
		assert(math.ceil(k) == k)

		# Split the timeserie
		data_split = np.split(data[modulo:], k)
		if modulo != 0:
			data_split.insert(0, list(data[:window_size]))
		data_split = np.asarray(data_split)

		return data_split