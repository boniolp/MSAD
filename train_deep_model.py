########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : train_deep_model
#
########################################################################

import argparse
import os
import re
import json

import numpy as np
import pandas as pd

from utils.train_deep_model_utils import ModelExecutioner
# from utils.evaluator import Evaluator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from utils.metrics_loader import MetricsLoader
from utils.config import *

from models.model.convnet import ConvNet
from models.model.inception_time import InceptionModel
from models.model.resnet import ResNetBaseline
from models.model.sit import SignalTransformer


# Dict of model names to Constructors
models = {
	'convnet':ConvNet,
	'inception_time':InceptionModel,
	'inception':InceptionModel,
	'resnet':ResNetBaseline,
	'sit':SignalTransformer,
}


def train_deep_model(
	data_path,
	model_name,
	split_per,
	seed,
	read_from_file,
	batch_size,
	model_parameters_file,
	epochs,
):

	# Setup for cmd line args and device
	device = 'cuda'
	window_size = int(re.search(r'\d+', str(args.path)).group())

	# Load the splits
	train_set, val_set, test_set = create_splits(
		data_path,
		split_per=split_per,
		seed=seed,
		read_from_file=read_from_file,
	)
	# For testing
	train_set, val_set, test_set = train_set[:50], val_set[:10], test_set[:10]

	# Load the data
	print('----------------------------------------------------------------')
	training_data = TimeseriesDataset(data_path, fnames=train_set)
	val_data = TimeseriesDataset(data_path, fnames=val_set)
	test_data = TimeseriesDataset(data_path, fnames=test_set)
	
	# Create the data loaders
	training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

	# Compute class weights to give them to the loss function
	class_weights = training_data.get_weights_subset(device)

	# Find the model's dimensionality for the scheduler
	model_parameters = json_file(model_parameters_file)
	
	# Create the model, load it on GPU and print it
	model = models[model_name.lower()](**model_parameters).to(device)
	
	# Create the executioner object
	model_name = model_parameters_file.split('/')[-1].replace('.json', '')
	model_name = "{}_{}".format(model_name.replace('.json', ''), window_size)

	model_execute = ModelExecutioner(
		model=model,
		model_name=model_name,
		device=device,
		criterion=nn.CrossEntropyLoss(weight=class_weights).to(device),
		runs_dir='results/runs/',
		weights_dir='results/weights/',
		learning_rate=0.00001
	)

	# Check device of torch
	model_execute.torch_devices_info()

	# Run training procedure
	model, results = model_execute.train(
			n_epochs=args.epochs, 
			training_loader=training_loader, 
			validation_loader=validation_loader, 
			verbose=True,
	)

	df = pd.DataFrame.from_dict(results, orient="index")
	if args.unsupervised is not None:
			unsupervised_name = str(args.unsupervised).split('/')[-1][:-4].replace('unsupervised_', '')
			df.to_csv(os.path.join('results', f"unsupervised_{model_name}_{unsupervised_name}.csv"))
	else:
			df.to_csv(os.path.join('results', model_name + '.csv'))

	if args.unsupervised is not None:
			# Evaluate model on test set
			metricsloader = MetricsLoader(TSB_new_metrics_path)
			metrics = metricsloader.get_names()
			test_fnames = [x[:-4] for x in test_set]
			evaluator = Evaluator()

			model_name = "{}_{}".format(args.params[1], window_size)
			for metric in metrics:
					metric_values = metricsloader.read(metric=metric).loc[test_fnames]
					curr_metrics = evaluator.compute_anom_score_simple(
							model=model,
							model_type="deep",
							fnames=test_set,
							metric_values=metric_values[detector_names],
							metric=metric,
							data_path=args.path,
							batch_size=args.batch
					)
					curr_metrics.columns = ["{}_{}".format(model_name, x) for x in curr_metrics.columns.values]

					# Save scores
					file_name = os.path.join("unsupervised_model_scores", metric, "{}_{}_{}.csv".format(model_name, metric, unsupervised_name))
					curr_metrics.to_csv(file_name)

def model_retriever(x):
	if not isinstance(x, str):
		raise argparse.ArgumentTypeError("{} not a string".format(x))

	try:
		model = models[x.lower()]
	except KeyError:
		raise argparse.ArgumentTypeError("{} not in accepted models".format(x))

	return model, x.lower()

def json_file(x):
	if not os.path.isfile(x):
		raise argparse.ArgumentTypeError("{} is not a file".format(x))

	try:
		with open(x) as f:
   			variables = json.load(f)
	except Exception as e:
		raise argparse.ArgumentTypeError("{} is not a json file".format(x))

	return variables


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='run_experiment',
		description='This function is made so that we can easily run configurable experiments'
	)
	
	parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
	parser.add_argument('-s', '--split', type=float, help='split percentage for train and val sets', default=0.7)
	parser.add_argument('-se', '--seed', type=int, default=None, help='Seed for train/val split')
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	parser.add_argument('-m', '--model', type=str, default='sit', help='model to run', required=True)
	parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
	parser.add_argument('-b', '--batch', type=int, help='batch size', default=64)
	parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=10)
	
	args = parser.parse_args()
	train_deep_model(
		data_path=args.path,
		split_per=args.split,
		seed=args.seed,
		read_from_file=args.file,
		model_name=args.model,
		model_parameters_file=args.params,
		batch_size=args.batch,
		epochs=args.epochs
	)
