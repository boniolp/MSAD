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

import torch
from torch.utils.data import DataLoader
from utils.metrics_loader import MetricsLoader

from utils.config import *
from utils.train_deep_model_utils import json_file
from utils.timeseries_dataset import read_files
from utils.evaluator import Evaluator


def eval_deep_model(data_path, model_name, model_path, model_parameters_file):
	"""Given a model and some data it predicts the time series given

	:param path_model:
	:param model_parameters_file:
	:param path_data:
	"""
	batch_size = 64
	window_size = int(re.search(r'\d+', str(data_path)).group())

	# Read models parameters
	model_parameters = json_file(model_parameters_file)

	# Load model
	model = deep_models[model_name](**model_parameters)
	model.load_state_dict(torch.load(model_path))
	model.to('cuda')

	# Read data
	fnames = read_files(data_path)

	# Evaluate model
	metricsloader = MetricsLoader(TSB_metrics_path)
	metrics = metricsloader.get_names()
	evaluator = Evaluator()

	model_name = "{}_{}".format(args.params[1], window_size)
	fnames = fnames[:10]
	preds = evaluator.predict(
		model,
		fnames,
		data_path,
		batch_size
	)

	print(preds)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='run_experiment',
		description='This function is made so that we can easily run configurable experiments'
	)
	
	parser.add_argument('-d', '--data', type=str, help='path to the time series to predict', required=True)
	parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
	parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', required=True)
	parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
	
	args = parser.parse_args()
	eval_deep_model(
		data_path=args.data, 
		model_name=args.model, 
		model_path=args.model_path, 
		model_parameters_file=args.params
	)