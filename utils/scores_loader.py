########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : scores_loader
#
########################################################################

import multiprocessing
import multiprocessing.pool as mpp

from utils.metrics_loader import MetricsLoader
from utils.data_loader import DataLoader
# from utils.config import *
from utils.metrics import generate_curve

from numba import jit
import os, glob
import time
from pathlib import Path
import warnings
from tqdm import tqdm
# import itertools, operator
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd


class ScoresLoader:
	def __init__(self, scores_path):
		self.scores_path = scores_path

	def get_detector_names(self):
		'''Returns the names of the detectors

		:return: list of strings
		'''
		detectors = []

		for dataset in os.listdir(self.scores_path):
			curr_detectors = []
			for name in os.listdir(os.path.join(self.scores_path, dataset)):
				curr_detectors.append(name)
			if len(detectors) < 1:
				detectors = curr_detectors.copy()
			elif not detectors == curr_detectors:
				raise ValueError('detectors are not the same in this dataset \'{}\''.format(dataset))
		detectors.sort()

		return detectors

	def load(self, file_names):
		'''
		Load the score for the specified files/timeseries

		:param dataset: list of files
		'''
		detectors = self.get_detector_names()
		scores = []
		idx_failed = []

		for i, name in enumerate(tqdm(file_names, desc='Loading scores')):
			name_split = name.split('/')[-2:]
			paths = [os.path.join(self.scores_path, name_split[0], detector, 'score', name_split[1]) for detector in detectors]
			data = []
			try:
				for path in paths:
					data.append(pd.read_csv(path, header=None).to_numpy())
			except Exception as e:
				idx_failed.append(i)
				continue
			scores.append(np.concatenate(data, axis=1))

		# Delete ts which failed to load
		if len(idx_failed) > 0:
			print('failed to load')
			for idx in sorted(idx_failed, reverse=True):
				print('\t\'{}\''.format(file_names[idx]))
				# del file_names[idx]

		return scores, idx_failed


	def write(self, file_names, detector, score, metric):
		'''Write some scores for a specific detector

		:param files_names: list of names (list of strings)
		:param detector: name of the detector (string)
		:param score: 1D arrays (as many as file names)
		'''
		for fname in file_names:
			dataset, ts_name = fname.split('/')[-2:]
			
			Path(os.path.join(self.scores_path, dataset, detector))\
			.mkdir(parents=True, exist_ok=True)
			Path(os.path.join(self.scores_path, dataset, detector, metric))\
			.mkdir(parents=True, exist_ok=True)

			# np.save(
			# 	os.path.join(self.scores_path, dataset, detector, metric, ts_name), 
			# 	score[:100])
			np.savetxt(
				os.path.join(self.scores_path, dataset, detector, metric, ts_name), 
				score, 
				fmt='%.2f', 
				delimiter='\n')
			

# -----------------------------------------------------
	
	@jit
	def compute_metric(self, labels, scores, metric, verbose=1, n_jobs=1):
		'''Computes desired metric for all labels and scores pairs.

		:param labels: list of arrays each representing the labels of a timeseries/sample
		:param scores: list of 2D arrays representing the scores of each detector on a
						timeseries/sample.
		:param metric: str, name of metric to produce
		:param verbose: to print or not to print info
		:return: metric values
		'''
		n_files = len(labels)
		results = []

		if len(labels) != len(scores):
			raise ValueError("length of labels and length of scores not the same")

		if scores[0].ndim == 1 or scores[0].shape[-1] == 1:
			args = [x + (metric,) for x in list(zip(labels, scores))]
			pool = multiprocessing.Pool(n_jobs)

			results = []
			for result in tqdm(pool.istarmap(self.compute_single_sample, args), total=len(args)):
				results.append(result)

			results = np.asarray([x.tolist() for x in results])
		else:
			for i, x, y in tqdm(zip(range(n_files), labels, scores), total=n_files, desc='Compute {}'.format(metric), disable=not verbose):
				results.append(self.compute_single_sample(x, y, metric))
			results = np.asarray(results)

		return results


	def compute_single_sample(self,	label, score, metric):
		'''Compute a metric for a single sample and multiple scores.

		:param label: 1D array of 0, 1 labels, (len_ts)
		:param score: 2D array, (len_ts, n_det)
		:param metric: string to which metric to compute
		:return: an array of values, one for each score
		'''
		if label.shape[0] != score.shape[0]:
			raise ValueError("label and score first dimension do not match. {} != {}".format(label.shape[0], score.shape[0]))

		if label.ndim > 1:
			raise ValueError("label has more dimensions than expected.")

		tick = time.process_time()
		result = np.apply_along_axis(self.compute_single_metric, 0, score, label, metric)
		# result = parallel_apply_along_axis(self.compute_single_metric, 0, score, label, metric)
		# print(type(result), result.shape)

		# fig, ax = plt.subplots(3, 4, figsize=(15, 10))
		# best = np.argmax(result)
		# x = np.linspace(0, label.shape[0], label.shape[0])
		# for i, axis in enumerate(ax):
		# 	for j, axs in enumerate(axis):
		# 		if i*4 + j == best:
		# 			axs.patch.set_alpha(0.3)
		# 			axs.patch.set_facecolor('green')
		# 		axs.title.set_text('{:.1f}%'.format(result[i*4 + j]*100))
		# 		axs.set_xlim([0, x[-1]])
		# 		axs.set_ylim([0, 1])
		# 		axs.plot(label, label='label', color='k', linewidth=2)
		# 		axs.plot(score[:, i*4 + j], label='score')
		# 		axs.legend()
		# 		axs.fill_between(x, label, score[:, i*4 + j])
		# 		plt.tight_layout()
		# plt.show()

		return result.T

	@jit
	def estimate_length(self, label):
		max_len = 0
		counter = 0

		for val in label:
			if val:
				counter += 1
			else:
				max_len = counter if counter > max_len else max_len
				counter = 0
		
		return max_len if max_len > 10 else 10

	def compute_single_metric(self, score, label, metric):
		'''Compute a metric for a single sample and score.

		:param label: 1D array of 0, 1 labels
		:param score: 1D array same length as label
		:param metric: string to which metric to compute
		:return: a single value
		'''
		if label.shape != score.shape:
			raise ValueError("label and metric should have the same length.")
		
		metric = metric.lower()
		if metric == 'naive':
			combined = np.vstack((label, score)).T
			diff = np.abs(np.diff(combined))
			result = 1 - np.mean(diff)
		elif np.all(0 == label):
			fpr, tpr, thresholds = metrics.roc_curve(label, score)
			thresholds[0] = 1
			result = 1 - metrics.auc(thresholds, fpr)
		elif np.all(1 == label):
			fpr, tpr, thresholds = metrics.roc_curve(label, score)
			thresholds[0] = 1
			result = metrics.auc(thresholds, tpr)
		elif metric == 'fscore':
			thresholds = np.linspace(1, 0, 20)			
			fscores = [self.compute_fscore(x, score, label) for x in thresholds]
			result = metrics.auc(thresholds, fscores)
		elif metric == 'auc_roc':
			fpr, tpr, _ = metrics.roc_curve(label, score)
			result = metrics.auc(fpr, tpr)
		elif metric == 'auc_pr':
			precision, recall, _ = metrics.precision_recall_curve(label, score)
			result = metrics.auc(recall, precision)
		elif metric == 'vus_roc':
			result, _ = generate_curve(label, score, 2*10)
		elif metric == 'vus_pr':
			_, result = generate_curve(label, score, 2*10)
		elif metric == 'vus':
			result = generate_curve(label, score, 2*10)
		else:
			raise ValueError("can't recognize metric requested")

		# result = round(result, 5)
		# if result > 1 or result < 0:
		# 	print('>> Pottentially faulty metrics result {}'.format(result))

		return result

	def compute_fscore(self, threshold, score, label):
		score = score > threshold
		return metrics.f1_score(label, score)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
	"""
	Like numpy.apply_along_axis(), but takes advantage of multiple
	cores.
	"""        
	# Effective axis where apply_along_axis() will be applied by each
	# worker (any non-zero axis number would work, so as to allow the use
	# of `np.array_split()`, which is only done on axis 0):
	effective_axis = 1 if axis == 0 else axis
	if effective_axis != axis:
		arr = arr.swapaxes(axis, effective_axis)

	# Chunks for the mapping (only a few chunks):
	chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
		for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

	pool = multiprocessing.Pool()
	individual_results = pool.map(unpacking_apply_along_axis, chunks)
	# Freeing the workers:
	pool.close()
	pool.join()

	return np.concatenate(individual_results)

def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap