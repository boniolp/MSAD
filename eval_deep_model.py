########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : eval_deep_model
#
########################################################################

import argparse
import re
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader

from utils.config import *
from utils.train_deep_model_utils import json_file
from utils.timeseries_dataset import read_files, create_splits
from utils.evaluator import Evaluator


def eval_deep_model(
	data_path, 
	model_name, 
	model_path=None, 
	model_parameters_file=None, 
	path_save=None, 
	fnames=None,
	read_from_file=None,
	model=None
):
	"""Given a model and some data it predicts the time series given

	:param data_path:
	:param model_name:
	:param model_path:
	:param model_parameters_file:
	:param path_save:
	:param fnames:
	"""
	window_size = int(re.search(r'\d+', str(data_path)).group())
	batch_size = 128

	assert(
		(model is not None) or \
		(model_path is not None and model_parameters_file is not None)
	), "You should provide the model or the path to the model, not both"

	assert(
		not (fnames is not None and read_from_file is not None)
	), "You should provide either the fnames or the path to the specific splits, not both"

	# Load the model only if not provided
	if model == None:
		# Read models parameters
		model_parameters = json_file(model_parameters_file)

		# Load model
		model = deep_models[model_name](**model_parameters)
		model.load_state_dict(torch.load(model_path))
		model.eval()
		model.to('cuda')	

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

	# Evaluate model
	evaluator = Evaluator()
	classifier_name = f"{model_name}_{window_size}"
	results = evaluator.predict(
		model=model,
		fnames=fnames,
		data_path=data_path,
		deep_model=True,
	)
	results = results.sort_index()
	results.columns = [f"{classifier_name}_{x}" for x in results.columns.values]
	
	# Print results
	print(results)
	counter = Counter(results[f"{model_name}_{window_size}_class"])
	print(dict(counter))
	
	# Save the results
	if path_save is not None:
		file_name = os.path.join(path_save, f"{classifier_name}_preds.csv")
		results.to_csv(file_name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='Evaluate deep learning models',
		description='Evaluate all deep learning architectures on a single or multiple time series \
			and save the results'
	)
	
	parser.add_argument('-d', '--data', type=str, help='path to the time series to predict', required=True)
	parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
	parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', required=True)
	parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the results', default="results/raw_predictions")
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)

	args = parser.parse_args()
	eval_deep_model(
		data_path=args.data, 
		model_name=args.model, 
		model_path=args.model_path, 
		model_parameters_file=args.params,
		path_save=args.path_save,
		read_from_file=args.file
	)