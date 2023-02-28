########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: models/oracles
# @file : morty
#
########################################################################

from utils.scores_loader import ScoresLoader

import numpy as np
import time


class Morty:
	'''Theoretical model that naively averages the scores of all detectors'''
	
	def fit(self, labels, scores, metric):
		morty_scores = []

		# Average the detectors' scores for every timeserie
		for score in scores:
			morty_scores.append(np.average(score, axis=1))

		# Create a scoresloader object
		scoresloader = ScoresLoader(scores_path='FAKEPATH')

		# Compute new score based on metric
		if not isinstance(metric, list):
			metric = [metric]

		metric_values_dict = {}
		# Check for VUS metrics
		is_vus = [x for x in metric if 'VUS' in x]
		if is_vus == metric:
			metric_values = scoresloader.compute_metric(labels, morty_scores, metric='vus')
			for i, curr_metric in enumerate(metric):
				metric_values_dict[curr_metric] = metric_values[:, i]
			return metric_values_dict

		for curr_metric in metric:
			metric_values_dict[curr_metric] = scoresloader.compute_metric(labels, morty_scores, metric=curr_metric)

		return metric_values_dict