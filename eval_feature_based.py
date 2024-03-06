########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : eval_feature_based
#
########################################################################

import argparse
import re
import os
from tqdm import tqdm
from time import perf_counter
from collections import Counter
import pandas as pd
from datetime import datetime

from utils.timeseries_dataset import read_files, create_splits
from utils.evaluator import Evaluator, load_classifier
from utils.config import *


def eval_feature_based(
	data_path, 
	model_name, 
	model_path, 
	path_save=None, 
	fnames=None,
	read_from_file=None,
):
	"""Predict some time series with the given rocket model

    :param data_path: Path to the data to predict.
    :param model_path: Path to the model to load and use for predictions.
    :param path_save: Path to save the evaluation results.
    :param fnames: List of file names (time series) to evaluate.
    :param read_from_file: File to read which time series to predict from a given path.
	"""
	window_size = int(re.search(r'\d+', str(data_path)).group())
	classifier_name = f"{model_name}_{window_size}" if str(window_size) not in model_name else model_name
	if read_from_file is not None and "unsupervised" in read_from_file:
		classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"
	all_preds = []
	inf_time = []

	# Load model 
	model = load_classifier(model_path)

	# Read data (single csv file or directory with csvs)
	data = pd.read_csv(data_path, index_col=0)
	labels, data = data['label'], data.drop('label', axis=1)

	# Load the splits
	if read_from_file is not None:
		_, val_set, test_set = create_splits(
			data_path,
			read_from_file=read_from_file,
		)
		fnames = test_set if len(test_set) > 0 else val_set
		fnames = [x[:-len(".csv")] for x in fnames]

	# if fnames is not defined then predict everything
	if fnames is None:
		data_index = list(data.index)
		fnames = list(set([tuple(x.rsplit('.', 1))[0] for x in data_index]))

	# Compute predictions and inference time
	for fname in tqdm(fnames, desc='Computing', unit='files'):
		# Load the data (already loaded just collecting them)
		x = data.filter(like=fname, axis=0)
		y = labels.filter(like=fname, axis=0)

		# Predict time series
		tic = perf_counter()
		preds = model.predict(x)
		counter = Counter(preds)
		most_voted = counter.most_common(1)
		toc = perf_counter()

		# Save info
		all_preds.append(detector_names[int(most_voted[0][0])])
		inf_time.append(toc-tic)
	results = pd.DataFrame(data=zip(all_preds, inf_time), columns=["class", "inf"], index=fnames)
	results.columns = [f"{classifier_name}_{x}" for x in results.columns.values]
	
	# Print results
	print(results)

	# Save the results
	if path_save is not None:
		file_name = os.path.join(path_save, f"{classifier_name}_preds.csv")
		results.to_csv(file_name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Evaluate feature based models',
		description='Evaluate feature based models \
			on a single or multiple time series and save the results'
	)
	
	parser.add_argument('-d', '--data', type=str, help='path to the time series to predict', required=True)
	parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
	parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', required=True)
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the results', default="results/raw_predictions")
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)

	args = parser.parse_args()
	eval_feature_based(
		data_path=args.data, 
		model_name=args.model,
		model_path=args.model_path,
		path_save=args.path_save,
		read_from_file=args.file,

	)