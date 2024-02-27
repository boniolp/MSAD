########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : Ensemble model anomaly detection
# @component: root
# @file : merge_scores
#
########################################################################

from utils.metrics_loader import MetricsLoader

import os
from utils.config import *
import pandas as pd
import argparse
import numpy as np

from natsort import natsorted, ns


def main(path, metric):
	df = None
	metricsloader = MetricsLoader(TSB_new_metrics_path)
	if metric not in metricsloader.get_names():
		raise ValueError(f"Not recognizable metric {metric}. Please use one of {metricsloader.get_names()}")

	# Read acc tables file and fix indexes to use later (if-else for VUS trick ;)
	acc_tables_files = [x for x in os.listdir(TSB_metrics_path) if metric in x]
	if len(acc_tables_files) != 1:
		acc_tables = pd.read_csv(os.path.join(TSB_metrics_path, 'mergedTable_AUC_PR.csv'))
		acc_tables.loc[:, detector_names] = np.NaN
	elif len(acc_tables_files) == 1:
		acc_tables = pd.read_csv(os.path.join(TSB_metrics_path, acc_tables_files[0]))
	else:
		raise ValueError(f'This file {acc_tables_files} should be unique please check it')

	acc_tables_filenames = acc_tables['filename']
	acc_tables_filenames = [x.replace('.txt', '.out') for x in acc_tables_filenames]
	acc_tables['filename'] = acc_tables_filenames
	acc_tables = acc_tables.set_index(keys=['dataset', 'filename'])

	# Read classifiers scores and fix indexes
	dir_path = os.path.join(path, metric)
	scores_files = [x for x in os.listdir(dir_path) if '.csv' in x]
	for file in scores_files:
		file_path = os.path.join(dir_path, file)
		tmp = pd.read_csv(file_path, index_col=0)
		
		if df is None:
			df = tmp
		else:
			df = pd.merge(df, tmp, left_index=True, right_index=True)
	df = df.reindex(natsorted(df.columns, key=lambda y: y.lower()), axis=1)

	# Add Genie and Morty
	genie = pd.read_csv(os.path.join(TSB_new_metrics_path, 'GENIE', f'{metric}.csv'), index_col=0)
	morty = pd.read_csv(os.path.join(TSB_new_metrics_path, 'MORTY', f'{metric}.csv'), index_col=0)
	genie_indexes = [x.replace(TSB_data_path+'/', '') for x in genie.index.tolist()]
	genie.index = genie_indexes
	oracles = pd.merge(genie, morty, left_index=True, right_index=True)
	df = pd.merge(df, oracles, left_index=True, right_index=True)
	
	# Change the indexes to dataset, filename
	old_indexes = df.index.tolist()
	old_indexes_split = [tuple(x.split('/')) for x in old_indexes]
	filenames_df = pd.DataFrame(old_indexes_split, index=old_indexes, columns=['dataset', 'filename'])
	df = pd.merge(df, filenames_df, left_index=True, right_index=True)
	df = df.set_index(keys=['dataset', 'filename'])

	# Merge the two dataframes now that they have common indexes
	print('Indexes not found in acc_tables file:')
	acc_tables_indexes = acc_tables.index
	df_indexes = df.index
	count = 0
	for df_index in df_indexes:
		if df_index not in acc_tables_indexes:
			print(count, df_index)
			count += 1	
	final_df = df.join(acc_tables)

	# Add the true labels into the final dataframe (labels come from AUC_PR only!)
	metrics_data = metricsloader.read('AUC_PR')
	metrics_data = metrics_data[detector_names]
	metrics_data = pd.merge(metrics_data, filenames_df, left_index=True, right_index=True)
	metrics_data = metrics_data.set_index(keys=['dataset', 'filename'])
	labels = metrics_data.idxmax(axis=1).to_frame(name='label')
	final_df = pd.merge(labels, final_df, left_index=True, right_index=True)

	# Find all sequences with missing detectors scores and fill them
	nan_values = final_df[final_df.isna().any(axis=1)]
	if 'VUS' in metric:
		nan_values_indexes = [os.path.join('/'.join(list(x))) for x in nan_values.index.tolist()]
	else:
		nan_values_indexes = [os.path.join('data', 'TSB', 'data', '/'.join(list(x))) for x in nan_values.index.tolist()]
	for detector in detector_names:
		detector_file_path = os.path.join(TSB_new_metrics_path, detector, f'{metric}.csv')
		detector_score = pd.read_csv(detector_file_path, index_col=0)
		missing_values = detector_score.loc[nan_values_indexes].to_numpy().reshape(-1)		
		
		counter = 0
		for index, row in nan_values.iterrows():
			final_df.at[index, detector] = missing_values[counter]
			counter += 1

	# Save the final dataframe
	print(final_df)
	final_df.to_csv(os.path.join(path, f'merged_scores_{metric}.csv'))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Merge scores',
		description="Merge all models' scores into one csv"
	)
	parser.add_argument('-p', '--path', help='path of the files to merge')
	parser.add_argument('-m', '--metric', help='metric to use')

	args = parser.parse_args()
	main(
		path=args.path, 
		metric=args.metric
	)