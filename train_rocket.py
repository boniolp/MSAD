########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : train_rocket
#
########################################################################

import argparse
import os
import re
from time import perf_counter
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.evaluator import save_classifier
from utils.config import *
from eval_rocket import eval_rocket



def run_rocket(data_path, split_per=0.7, seed=None, read_from_file=None, eval_model=False, path_save=None):
	# Set up
	window_size = int(re.search(r'\d+', data_path).group())
	classifier_name = f"rocket_{window_size}"
	training_stats = {}

	# Load the splits
	train_set, val_set, test_set = create_splits(
		data_path,
		split_per=split_per,
		seed=seed,
		read_from_file=read_from_file,
	)
	# train_set, val_set, test_set = train_set[:10], val_set[:10], test_set[:5]

	# Load the data
	training_data = TimeseriesDataset(data_path, fnames=train_set)
	val_data = TimeseriesDataset(data_path, fnames=val_set)
	test_data = TimeseriesDataset(data_path, fnames=test_set)

	# Split data from labels
	X_train, y_train = training_data.__getallsamples__().astype('float32'), training_data.__getalllabels__()
	X_val, y_val = val_data.__getallsamples__().astype('float32'), val_data.__getalllabels__()
	
	# Create the feature extractor, the scaler, and the classifier
	minirocket = MiniRocket(num_kernels=10000, n_jobs=-1)
	scaler = StandardScaler(with_mean=False, copy=False)
	clf = SGDClassifier(loss='log_loss', n_jobs=-1)

	tic = perf_counter()
	X_train = minirocket.fit_transform(X_train).to_numpy()
	print("minirocket fitted: {:.3f} secs".format(perf_counter()-tic))
	
	# Setup batching
	batch_size = 32768
	indexes = np.arange(X_train.shape[0])
	indexes_shuffled = shuffle(indexes)

	# Fit scaler
	for iterator_train in tqdm(range(0, X_train.shape[0], batch_size), desc='fitting-scaler'):
			curr_batch = indexes_shuffled[iterator_train:iterator_train+batch_size]
			X = X_train[curr_batch]
			scaler.partial_fit(X)

	# Transform the data in batches
	for iterator_train in tqdm(range(0, X_train.shape[0], batch_size), desc='transforming'):
			curr_batch = indexes_shuffled[iterator_train:iterator_train+batch_size]
			X_train[curr_batch] = scaler.transform(X_train[curr_batch])

	# Fit the classifier
	for iterator_train in tqdm(range(0, X_train.shape[0], batch_size), desc='training'):
			curr_batch = indexes_shuffled[iterator_train:iterator_train+batch_size]
			X = X_train[curr_batch]
			Y = y_train[curr_batch]
			clf.partial_fit(X, Y, classes=list(np.arange(12)))
	toc = perf_counter()

	# Put every fitted component into a pipeline
	classifier = make_pipeline(
			minirocket,
			scaler,
			clf
	)
	del X_train
	del y_train

	# Print training time
	training_stats["training_time"] = toc-tic
	print(f"training time: {training_stats['training_time']:.3f} secs")

	# Print valid accuracy and inference time
	tic = perf_counter()
	classifier_score = classifier.score(X_val, y_val)
	toc = perf_counter()
	training_stats["val_acc"] = classifier_score
	training_stats["avg_inf_time"] = ((toc-tic)/X_val.shape[0]) * 1000
	print(f"valid accuracy: {training_stats['val_acc']:.3%}")
	print(f"inference time: {training_stats['avg_inf_time']:.3} ms")

	# Save training stats
	timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
	df = pd.DataFrame.from_dict(training_stats, columns=["training_stats"], orient="index")
	df.to_csv(os.path.join(save_done_training, f"{classifier_name}_{timestamp}.csv"))

	# Save pipeline
	saving_dir = os.path.join(path_save, classifier_name) if classifier_name.lower() not in path_save.lower() else path_save
	saved_model_path = save_classifier(classifier, saving_dir, fname=None)

	# Evaluate on test set or val set
	if eval_model:
		eval_set = test_set if len(test_set) > 0 else val_set
		eval_rocket(
			data_path=data_path, 
			model_path=saved_model_path,
			path_save=path_save_results,
			fnames=eval_set,
		)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='train_rocket',
		description='Script for training the MiniRocket feature_extractor+classifier',
	)
	parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
	parser.add_argument('-sp', '--split_per', type=float, help='split percentage for train and val sets', default=0.7)
	parser.add_argument('-s', '--seed', type=int, help='seed for splitting train, val sets (use small number)', default=None)
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="results/weights")

	args = parser.parse_args()
	run_rocket(
		data_path=args.path,
		split_per=args.split_per,
		seed=args.seed,
		read_from_file=args.file,
		eval_model=args.eval_true,
		path_save=args.path_save,
	)
