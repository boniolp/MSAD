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
import silence_tensorflow.auto

import numpy as np
import pandas as pd

from utils.train_deep_model_utils import ModelExecutioner, json_file
# from utils.evaluator import Evaluator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from utils.metrics_loader import MetricsLoader
from utils.config import *


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

	# Read models parameters
	model_parameters = json_file(model_parameters_file)
	
	# Create the model, load it on GPU and print it
	model = deep_models[model_name.lower()](**model_parameters).to(device)
	
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

	# Save training stats (uncomment to keep track of finished trained models)
	# df = pd.DataFrame.from_dict(results, orient="index")
	# df.to_csv(os.path.join('results/done_training/', model_name + '.csv'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='run_experiment',
		description='This function is made so that we can easily run configurable experiments'
	)
	
	parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
	parser.add_argument('-s', '--split', type=float, help='split percentage for train and val sets', default=0.7)
	parser.add_argument('-se', '--seed', type=int, default=None, help='Seed for train/val split')
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
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
