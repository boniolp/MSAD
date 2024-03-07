########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : merge_unsupervised_scores
#
########################################################################

from utils.metrics_loader import MetricsLoader
from utils.config import TSB_metrics_path, TSB_data_path, detector_names, TSB_acc_tables_path

import os
import pandas as pd
import argparse
import numpy as np
from natsort import natsorted

def read_csv_with_substring(directory, substring):
    """
    Read all CSV files in a directory whose filenames contain a specified substring.

    Args:
    - directory (str): Path to the directory containing CSV files.
    - substring (str): Substring to search for in the filenames.

    Returns:
    - list: List of paths for CSV files containing the substring in their filenames.
    """
    # List to store paths of matching CSV files
    csv_files = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .csv extension and the substring in its filename
        if filename.endswith('.csv') and substring in filename:
            # Append the file path to the list
            csv_files.append(os.path.join(directory, filename))

    return csv_files


def merge_unsupervised_scores(path, metric, save_path):
	og_scores_path = os.path.join('results', 'unsupervised_results', metric.upper())

	# Load MetricsLoader object
	metricsloader = MetricsLoader(TSB_metrics_path)
	
	# Check if given metric exists
	if metric.upper() not in metricsloader.get_names():
		raise ValueError(f"Not recognizable metric {metric}. Please use one of {metricsloader.get_names()}")

	# Read original unsupervised scores file to draw AD columns (Anomaly Detectors NOT Model Selectors)
	og_results = []
	split_names = []
	for filename in os.listdir(og_scores_path):
		if filename.endswith('.csv'):
			split_names.append(filename[:-len('.csv')])
			og_results.append(pd.read_csv(os.path.join(og_scores_path, filename), index_col=0))

	# Read detectors and oracles scores
	metric_scores = metricsloader.read(metric.upper())
			
	# Read all MS per split
	for filename, og_res in zip(split_names, og_results):
		files = read_csv_with_substring(path, f'{filename}_preds')
		df = None

		# Read MS file
		for file in files:
			current_classifier = pd.read_csv(file, index_col=0)
			current_classifier = current_classifier.rename(columns=lambda x: x.replace(f'_{filename}', ''))
			
			col_name = [x for x in current_classifier.columns if "class" in x][0]
		
			values = np.diag(metric_scores.loc[current_classifier.index, current_classifier.iloc[:, 0]])
			curr_df = pd.DataFrame(values, index=current_classifier.index, columns=[col_name.replace("_class", "_score")])
			curr_df = pd.merge(curr_df, current_classifier, left_index=True, right_index=True)
			
			if df is None:
				df = curr_df
			else:
				df = pd.merge(df, curr_df, left_index=True, right_index=True)
		
		# Add Oracles, labels and Anomaly detectors stats on splits
		columns_to_keep = ['label', 'Oracle', 'Avg Ens', 'best_ad_train', 'average_ad_train', 'worst_ad_train', 'best_ad_test', 'average_ad_test', 'worst_ad_test']
		df = pd.merge(og_res[columns_to_keep], df, left_index=True, right_index=True)

		# Save the final dataframe
		df.to_csv(os.path.join(save_path, f'current_{filename}.csv'), index=True)
		print(df)
			
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Merge unsupervised scores',
		description="Merge all unsupervised models' scores into one csv"
	)
	parser.add_argument('-p', '--path', help='path of the files to merge')
	parser.add_argument('-m', '--metric', help='metric to use')
	parser.add_argument('-s', '--save_path', help='where to save the result')
	# parser.add_argument('-time', '--time-true', action="store_true", help='whether to produce time results')


	args = parser.parse_args()
	
	merge_unsupervised_scores(
        path=args.path, 
        metric=args.metric,
        save_path=args.save_path,
    )
