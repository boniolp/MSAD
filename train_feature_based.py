########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : train_feature_based
#
########################################################################


import argparse
import os
from time import perf_counter
import re
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils.timeseries_dataset import create_splits, TimeseriesDataset

from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.scores_loader import ScoresLoader
from utils.config import *

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC

names = {
		"knn": "Nearest Neighbors",
		"svc_linear": "Linear SVM",
		"decision_tree": "Decision Tree",
		"random_forest": "Random Forest",
		"mlp": "Neural Net",
		"ada_boost": "AdaBoost",
		"bayes": "Naive Bayes",
		"qda": "QDA",
}

classifiers = {
		"knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
		"svc_linear": LinearSVC(C=0.025, verbose=True),
		"decision_tree": DecisionTreeClassifier(max_depth=5),
		"random_forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1, verbose=True),
		"mlp": MLPClassifier(alpha=1, max_iter=1000, verbose=True),
		"ada_boost": AdaBoostClassifier(),
		"bayes": GaussianNB(),
		"qda": QuadraticDiscriminantAnalysis(),
}


def train_feature_based(data_path, classifier_name, split_per=0.7, seed=None, read_from_file=None, eval_model=False):
	# Set up
	window_size = int(re.search(r'\d+', data_path).group())
	
	# Load the splits
	train_set, val_set, test_set = create_splits(
		data_path,
		split_per=split_per,
		seed=seed,
		read_from_file=read_from_file,
	)
	train_indexes = [x[:-4] for x in train_set]
	val_indexes = [x[:-4] for x in val_set]
	test_indexes = [x[:-4] for x in test_set]

	# Read tabular data
	data = pd.read_csv(data_path, index_col=0)

	# Reindex them
	data_index = list(data.index)
	new_index = [tuple(x.rsplit('.', 1)) for x in data_index]
	new_index = pd.MultiIndex.from_tuples(new_index, names=["name", "n_window"])
	data.index = new_index
	
	# Create subsets
	training_data = data.loc[data.index.get_level_values("name").isin(train_indexes)]
	val_data = data.loc[data.index.get_level_values("name").isin(val_indexes)]
	test_data = data.loc[data.index.get_level_values("name").isin(test_indexes)]
	
	# Split data from labels
	y_train, X_train = training_data['label'], training_data.drop('label', 1)
	y_val, X_val = val_data['label'], val_data.drop('label', 1)
	y_test, X_test = test_data['label'], test_data.drop('label', 1)

	# Create the names of the files we are going to use for evaluation
	# val_set = [re.sub(r'\d+$', '', x) for x in list(val_data.index)]
	# val_set = [x[:-1] for x in list(set(val_set))]

	# Select the classifier
	classifier = classifiers[classifier_name]

	# For svc_linear use only a random subset of the dataset to train
	if 'svc' in classifier_name and len(y_train) > 200000:
		rand_ind = np.random.randint(low=0, high=len(y_train), size=200000)
		X_train = X_train.iloc[rand_ind]
		y_train = y_train.iloc[rand_ind]

	# Fit the classifier
	print('----------------------------------')
	print(f'Training {names[classifier_name]}...')
	tic = perf_counter()
	classifier.fit(X_train, y_train)
	toc = perf_counter()
	print("training time: {:.3f} secs".format(toc-tic))
	tic = perf_counter()
	classifier_score = classifier.score(X_val, y_val)
	toc = perf_counter()
	print('valid accuracy: {:.3%}'.format(classifier_score))
	print("inference time: {:.3} ms".format(((toc-tic)/X_val.shape[0]) * 1000))

	# Load metrics
	metricsloader = MetricsLoader(TSB_metrics_path)
	metrics = metricsloader.get_names()

	if not eval_model:
		return
	elif len(test_indexes) == 0:
		raise ValueError('No test set given for evaluating the model')

	# Evaluating the model
	# test_indexes = test_indexes[:10]
	for metric in metrics:
		metric_values = metricsloader.read(metric=metric).loc[test_indexes][detector_names]
		inf_time = []
		all_preds = []
		pred_scores = []
		for fname in tqdm(test_indexes, desc=f'Computing {metric}'):
			# Load the data (already loaded just collecting them)
			x = X_test.filter(like=fname, axis=0)
			y = y_test.filter(like=fname, axis=0)

			# Predict time series
			tic = perf_counter()
			preds = classifier.predict(x)
			counter = Counter(preds)
			most_voted = counter.most_common(1)
			toc = perf_counter()

			# Save info
			inf_time.append(toc-tic)
			all_preds.append(detector_names[int(most_voted[0][0])])
			pred_scores.append(metric_values.loc[fname].iloc[int(most_voted[0][0])])

		# Create df
		curr_metrics = pd.DataFrame(data=zip(pred_scores, all_preds, inf_time), columns=["score", "class", "inf"], index=test_indexes)
		curr_metrics.columns = ["{}_{}_{}".format(classifier_name, str(window_size), x) for x in curr_metrics.columns.values]

		# Print results
		print(curr_metrics)

		# Save scores file (already exist in demo but uncomment if you want to reproduce them - fill in PATH)
		# model_name = '_'.join([classifier_name, str(window_size)])
		# file_name = os.path.join("PATH", metric, "{}_{}.csv".format(model_name, metric))
		# curr_metrics.to_csv(file_name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='train_feature_based',
		description='Script for training the traditional classifiers',
	)
	parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
	parser.add_argument('-c', '--classifier', type=str, help='classifier to run', required=True)
	parser.add_argument('-sp', '--split_per', type=float, help='split percentage for train and val sets', default=0.7)
	parser.add_argument('-s', '--seed', type=int, help='seed for splitting train, val sets (use small number)', default=None)
	parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	# parser.add_argument('-e', '--eval', type=bool, help='whether to evaluate the model on test data after training', default=False)
	parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')

	args = parser.parse_args()

	# Option to all classifiers
	if args.classifier == 'all':
		clf_list = list(classifiers.keys())
	else:
		clf_list = [args.classifier]

	for classifier in clf_list:
		train_feature_based(
			data_path=args.path, 
			classifier_name=classifier,
			split_per=args.split_per, 
			seed=args.seed,
			read_from_file=args.file,
			eval_model=args.eval_true
		)
