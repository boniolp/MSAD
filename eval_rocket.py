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

from utils.timeseries_dataset import read_files
from utils.evaluator import Evaluator, load_classifier


def eval_rocket(data_path, model_path, path_save=None, fnames=None):
	"""Predict some time series with the given rocket model

	:param data_path: path to the data to predict
	:param model_path: path to the model to load and use for predictions
	:param read_from_file: file to read which time series to predict from a given path
	:param data: data to call directly from another function with loaded data
	"""
	window_size = int(re.search(r'\d+', str(data_path)).group())
	classifier_name = f"rocket_{window_size}"

	# Load model 
	model = load_classifier(model_path)

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



if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Evaluate rocket models',
		description='Evaluate rocekt models \
			on a single or multiple time series and save the results'
	)
	
	parser.add_argument('-d', '--data', type=str, help='path to the time series to predict', required=True)
	parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', required=True)
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the results', default="results/raw_predictions")

	args = parser.parse_args()
	eval_rocket(
		data_path=args.data, 
		model_path=args.model_path,
		path_save=args.path_save,
	)