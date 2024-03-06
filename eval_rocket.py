########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : eval_rocket
#
########################################################################

import argparse
import re
import os
from datetime import datetime

from utils.timeseries_dataset import read_files, create_splits
from utils.evaluator import Evaluator, load_classifier


def eval_rocket(data_path, model_path, path_save=None, fnames=None, read_from_file=None):
	"""Predict time series with the given ROCKET model.

	:param data_path: Path to the data to predict.
	:param model_path: Path to the model to load and use for predictions.
	:param path_save: Path to save the evaluation results.
	:param fnames: List of file names (time series) to predict.
	:param read_from_file: File to read which time series to predict from a given path.

	Returns:
	DataFrame: A DataFrame containing the predicted time series.
	"""
	window_size = int(re.search(r'\d+', str(data_path)).group())
	classifier_name = f"rocket_{window_size}"
	if read_from_file is not None and "unsupervised" in read_from_file:
		classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"
	elif "testsize_" in model_path:
		extra = model_path.split('/')[-2].replace(classifier_name, "")
		classifier_name += extra
		
	assert(
		not (fnames is not None and read_from_file is not None)
	), "You should provide either the fnames or the path to the specific splits, not both"

	# Load model 
	model = load_classifier(model_path)

	# Load the splits
	if read_from_file is not None:
		_, val_set, test_set = create_splits(
			data_path,
			read_from_file=read_from_file,
		)
		fnames = test_set if len(test_set) > 0 else val_set
		# fnames = fnames[:100]
	else:
		# Read data (single csv file or directory with csvs)
		if '.csv' == data_path[-len('.csv'):]:
			tmp_fnames = [data_path.split('/')[-1]]
			data_path = data_path.split('/')[:-1]
			data_path = '/'.join(data_path)
		else:
			tmp_fnames = read_files(data_path)

		# Keep specific time series if fnames is given
		if fnames is not None:
			fnames_len = len(fnames)
			fnames = [x for x in tmp_fnames if x in fnames]
			if len(fnames) != fnames_len:
				raise ValueError("The data path does not include the time series in fnames")
		else:
			fnames = tmp_fnames

	# Compute predictions and inference time
	evaluator = Evaluator()
	results = evaluator.predict(
		model=model,
		fnames=fnames,
		data_path=data_path,
		deep_model=False,
	)
	results.columns = [f"{classifier_name}_{x}" for x in results.columns.values]
	
	# Print results
	print(results)

	# Save the results
	if path_save is not None:
		file_name = os.path.join(path_save, f"{classifier_name}_preds.csv")
		results.to_csv(file_name)

	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Evaluate rocket models',
		description='Evaluate rocekt models \
			on a single or multiple time series and save the results'
	)
	
	parser.add_argument('-d', '--data', type=str, help='path to the time series to predict', required=True)
	parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', required=True)
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the results', default="results/raw_predictions")
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)

	args = parser.parse_args()
	eval_rocket(
		data_path=args.data, 
		model_path=args.model_path,
		path_save=args.path_save,
		read_from_file=args.file
	)