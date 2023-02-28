########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: models
# @file : avg_ens
#
########################################################################

from utils.scores_loader import ScoresLoader

import numpy as np
import copy


class Avg_ens:
	'''Theoretical model that naively averages the scores of all detectors'''
	
	def fit(self, labels, scores, metrics, n_jobs=1):
		'''Computes all metrics for the Average Ensemble model. If both VUS_ROC
		and VUS_PR are requested a trick is done to compute them in parallel
		and save time.

		:param labels: anomaly labels of the timeseries to compute
		:param scores: the already computed scores of the different detectors
		:param metrics: metrics to compute (['AUC_ROC', 'AUC_PR', 'VUS_ROC', 'VUS_PR'])
		:param n_jobs: Threads to use in parallel to compute the metrics faster
		:return metric_values_dict: a dictionary with all the computed metrics
		'''
		avg_ens_scores = []
		metric = copy.deepcopy(metrics)

		# Average the detectors' scores for every timeserie
		for score in scores:
			avg_ens_scores.append(np.average(score, axis=1))

		# Create a scoresloader object
		scoresloader = ScoresLoader(scores_path='DUMMYPATH')

		# Compute new score based on metric
		if not isinstance(metric, list):
			metric = [metric]
		metric_values_dict = {}

		# Compute VUS metrics in parallel if both VUS_ROC & VUS_PR are requested
		if 'VUS_ROC' in metric and 'VUS_PR' in metric:
			metric.remove('VUS_ROC')
			metric.remove('VUS_PR')
			metric.append('vus')
		
		for curr_metric in metric:
			if curr_metric == 'vus':
				metric_values = scoresloader.compute_metric(labels, avg_ens_scores, metric='vus', n_jobs=n_jobs)
				for i, m in enumerate(['VUS_ROC', 'VUS_PR']):
					metric_values_dict[m] = metric_values[:, i]
			else:
				metric_values_dict[curr_metric] = scoresloader.compute_metric(labels, avg_ens_scores, metric=curr_metric, n_jobs=n_jobs)

		return metric_values_dict