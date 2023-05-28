########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: models
# @file : oracle
#
########################################################################

from utils.metrics_loader import MetricsLoader
from utils.config import * 

import numpy as np
import re


class Oracle:
	'''Theoretical model, a.k.a. the Oracle. The model can simulate	
	theoretical results for different accuracy values and randomness modes.
	The modes of randomness are the following:
	- true: whenever you do a mistake the mistake is actually random
	- lucky: whenever you do a mistake you select the 2nd best detector
				for this time series
	- unlucky: whenever you do a mistake you select the worst detector
				for this time series
	- best-k: whenever you do a mistake you select the k best detector
				for this time series (e.g. lucky is equal to best-2)
	'''
	
	def __init__(self, metrics_path, acc, randomness='true'):
		'''When an object of this oracle is created, the path to the metrics,
		the accuracy that should be simulated and the randomness modes
		should be given.
		'''
		self.path = metrics_path
		self.acc = acc
		self.randomness = randomness

	def fit(self, metric):
		''' Create the results of the Oracle according to the hyper-parameters
		of the object

		:param metric: the evaluation measure that will be returned
		:return fnames: the names of the files that were processed
		:return score: the values of the evaluation measures computed per time series
		'''

		# Create MetricsLoader object
		metricsloader = MetricsLoader(self.path)

		# Read metric's values
		data = metricsloader.read(metric=metric)
		data = data[detector_names]
		fnames = data.index
		data = data.to_numpy()

		# Flip a coin for every timeserie according to the accuracy
		coin = np.random.choice([True, False], data.shape[0], p=[self.acc, 1 - self.acc])
		score = np.zeros(data.shape[0])
		argmax = np.argmax(data, axis=1)
		inv_coin = np.invert(coin)
	
		# Create a 2d array that includes all detectors except the correct
		except_argmax = list([np.arange(0, data.shape[1])]) * data.shape[0]
		except_argmax = np.stack(except_argmax, axis=0)
		mask_2d = np.ones(data.shape, dtype=bool)
		mask_2d[np.arange(mask_2d.shape[0]), argmax] = False
		except_argmax = except_argmax[mask_2d].reshape(data.shape[0], -1)
		random_choices = except_argmax[np.arange(except_argmax.shape[0]), np.random.choice(except_argmax.shape[1], except_argmax.shape[0])]

		score[coin] = data[coin, argmax[coin]]
		if self.randomness == 'true':
			score[inv_coin] = data[inv_coin, random_choices[inv_coin]]
		else:
			data.sort(axis=1)
			if self.randomness == 'lucky':
				score[inv_coin] = data[inv_coin, -2]
			elif self.randomness == 'unlucky':
				score[inv_coin] = data[inv_coin, 0]
			elif 'best' in self.randomness:
				pick_best = int(re.search(r'\d+', self.randomness).group())
				score[inv_coin] = data[inv_coin, -pick_best]
			else:
				raise ValueError(f"randomness {self.randomness} not valid")



		return fnames, score