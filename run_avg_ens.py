########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : run_avg_ens
#
########################################################################


from models.model.avg_ens import Avg_ens

from utils.scores_loader import ScoresLoader
from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.config import *

import argparse
import numpy as np
import sys


def create_avg_ens(n_jobs=1):
	'''Create, fit and save the results for the 'Avg_ens' model

	:param n_jobs: Threads to use in parallel to compute the metrics faster
	'''

	# Load metrics' names
	metricsloader = MetricsLoader(TSB_metrics_path)
	metrics = metricsloader.get_names()

	# Load data
	dataloader = DataLoader(TSB_data_path)
	datasets = dataloader.get_dataset_names()
	x, y, fnames = dataloader.load(datasets)

	# Load scores
	scoresloader = ScoresLoader(TSB_scores_path)
	scores, idx_failed = scoresloader.load(fnames)

	# Remove failed idxs
	if len(idx_failed) > 0:
		for idx in sorted(idx_failed, reverse=True):
			del x[idx]
			del y[idx]
			del fnames[idx]

	# Create Avg_ens
	avg_ens = Avg_ens()
	metric_values = avg_ens.fit(y, scores, metrics, n_jobs=n_jobs)
	for metric in metrics:
		# Write metric values for avg_ens
		metricsloader.write(metric_values[metric], fnames, 'AVG_ENS', metric)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='run_avg_ense',
		description="Create the average ensemble model"
	)
	parser.add_argument('-n', '--n_jobs', type=int, default=4,
		help='Threads to use for parallel computation'
	)
	args = parser.parse_args()

	create_avg_ens(n_jobs=args.n_jobs)