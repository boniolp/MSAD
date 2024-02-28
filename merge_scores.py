########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : merge_scores
#
########################################################################

from utils.metrics_loader import MetricsLoader
from utils.config import TSB_metrics_path, TSB_data_path, detector_names, TSB_acc_tables_path

import os
import pandas as pd
import argparse
import numpy as np
from natsort import natsorted


def merge_scores(path, metric, save_path):
	# Load MetricsLoader object
	metricsloader = MetricsLoader(TSB_metrics_path)
	
	# Check if given metric exists
	if metric.upper() not in metricsloader.get_names():
		raise ValueError(f"Not recognizable metric {metric}. Please use one of {metricsloader.get_names()}")

	# Read accuracy table, fix indexing, remove detectors scores
	acc_tables_path = os.path.join(TSB_acc_tables_path, f"mergedTable_AUC_PR.csv")
	acc_tables = pd.read_csv(acc_tables_path, index_col=['dataset', 'filename'])
	acc_tables = acc_tables.drop(columns=detector_names)

	# Read detectors and oracles scores
	metric_scores = metricsloader.read(metric.upper())
	
	# Read classifiers predictions, and add scores
	df = None
	scores_files = [x for x in os.listdir(path) if '.csv' in x]
	for file in scores_files:
		file_path = os.path.join(path, file)
		current_classifier = pd.read_csv(file_path, index_col=0)
		col_name = [x for x in current_classifier.columns if "class" in x][0]
		
		values = np.diag(metric_scores.loc[current_classifier.index, current_classifier.iloc[:, 0]])
		curr_df = pd.DataFrame(values, index=current_classifier.index, columns=[col_name.replace("_class", "")])
		curr_df = pd.merge(current_classifier[col_name], curr_df, left_index=True, right_index=True)
		
		if df is None:
			df = curr_df
		else:
			df = pd.merge(df, curr_df, left_index=True, right_index=True)
	df = df.reindex(natsorted(df.columns, key=lambda y: y.lower()), axis=1)

	# Add Oracle (TRUE_ORACLE-100) and Averaging Ensemble
	df = pd.merge(df, metric_scores[["TRUE_ORACLE-100", "AVG_ENS"] + detector_names], left_index=True, right_index=True)
	df.rename(columns={'TRUE_ORACLE-100': 'Oracle', 'AVG_ENS': 'Avg Ens'}, inplace=True)
	
	# Add true labels from AUC_PR metrics
	auc_pr_detectors_scores = metricsloader.read('AUC_PR')[detector_names]
	labels = auc_pr_detectors_scores.idxmax(axis=1).to_frame(name='label')
	df = pd.merge(labels, df, left_index=True, right_index=True)
	
	# Change the indexes to dataset, filename
	old_indexes = df.index.tolist()
	old_indexes_split = [tuple(x.split('/')) for x in old_indexes]
	filenames_df = pd.DataFrame(old_indexes_split, index=old_indexes, columns=['dataset', 'filename'])
	df = pd.merge(df, filenames_df, left_index=True, right_index=True)
	df = df.set_index(keys=['dataset', 'filename'])

	# Merge the two dataframes now that they have common indexes
	final_df = df.join(acc_tables)
	indexes_not_found = final_df[final_df.iloc[:, -len(acc_tables.columns):].isna().any(axis=1)].index.tolist()
	print('Indexes not found in acc_tables file:')
	[print(i, x) for i, x in enumerate(indexes_not_found)]

	# Save the final dataframe
	final_df.to_csv(os.path.join(save_path, f'all_acuracy_{metric.upper()}.csv'), index=True)
	print(final_df)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Merge scores',
		description="Merge all models' scores into one csv"
	)
	parser.add_argument('-p', '--path', help='path of the files to merge')
	parser.add_argument('-m', '--metric', help='metric to use')
	parser.add_argument('-s', '--save_path', help='where to save the result')
	args = parser.parse_args()
	
	merge_scores(
		path=args.path, 
		metric=args.metric,
		save_path=args.save_path,
	)