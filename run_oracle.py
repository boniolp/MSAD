########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : run_oracle
#
########################################################################

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted, ns
import seaborn as sns
from tqdm import tqdm

from models.model.oracle import Oracle
from utils.metrics_loader import MetricsLoader
# from utils.config import *


def create_oracle(path, acc=1, randomness='true'):

	if acc > 1:
		raise ValueError("Accuracy can not be bigger than 1")
	if randomness not in ['true', 'lucky', 'unlucky'] + [f'best-{k}' for k in range(2, 13)]:
		raise ValueError(f"Randomness can not be {randomness}")

	# Create the oracle
	model = Oracle(metrics_path=path, acc=acc, randomness=randomness)

	# Create metrics object and get metrics' names
	metricsloader = MetricsLoader(path)
	metrics = metricsloader.get_names()

	# Fit it to the data
	for metric in metrics:
		files_names, score = model.fit(metric=metric)

		# Write new score
		acc_str = str(int(round(acc, 2) * 100))
		name = randomness.upper() + '_' + 'ORACLE-' + acc_str
		metricsloader.write(score, files_names, name, metric)


def eval_oracle(path):
	oracles = [os.path.join(path, x) for x in os.listdir(path) if 'ORACLE-' in x]
	# first_oracle = pd.read_csv(os.path.join(path, 'ORACLE', 'AUC_PR.csv'), index_col=0)
	# oracle_indexes = [x.replace(TSB_data_path+'/', '') for x in first_oracle.index.tolist()]
	# first_oracle.index = oracle_indexes
	# first_oracle = first_oracle.sort_index()

	all_oracles = None
	for oracle in oracles:
		files = [os.listdir(oracle)]
		df = pd.read_csv(os.path.join(oracle, 'AUC_PR.csv'), index_col=0)

		if all_oracles is None:
			# all_oracles = pd.merge(first_oracle, df, left_index=True, right_index=True)
			all_oracles = df
		else:
			all_oracles = pd.merge(all_oracles, df, left_index=True, right_index=True)

	# all_oracles = all_oracles.reindex(natsorted(all_oracles.columns, key=lambda y: y.lower()), axis=1)
	order = list(all_oracles.median().sort_values().index)
	
	# Create boxplot
	plt.figure(figsize=(19, 12))
	sns.boxplot(order=order,
				data=all_oracles)
	# plt.grid(visible=True)
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='run_oracle',
		description="Create the oracle with the accuracy value you\
		want to simulate"
	)
	parser.add_argument('-p', '--path', type=str, required=True,
		help='Path to metrics'
	)
	parser.add_argument('-a', '--acc', type=str, default=1.0, 
		help='The accuracy that you want to simulate'
	)
	parser.add_argument('-r', '--randomness', type=str, default='true',
		help='The randomness that you want to simulate'
	)
	args = parser.parse_args()

	# Run single
	if args.acc != 'all':
		create_oracle(
			path=args.path, 
			acc=int(args.acc), 
			randomness=args.randomness
		)
	elif args.acc == 'all':
		acc_all = [1.0, 0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.00]
		
		for acc in tqdm(acc_all, desc='Oracle'):
			create_oracle(
			path=args.path, 
			acc=acc, 
			randomness=args.randomness
		)

	# Evaluate with boxplot
	eval_oracle(path=args.path)