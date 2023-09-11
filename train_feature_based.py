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
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from eval_feature_based import eval_feature_based
from utils.evaluator import save_classifier
from utils.config import *

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


def train_feature_based(data_path, classifier_name, split_per=0.7, seed=None, read_from_file=None, eval_model=False, path_save=None):
	# Set up
	window_size = int(re.search(r'\d+', data_path).group())
	training_stats = {}
	original_dataset = data_path.split('/')[:-1]
	original_dataset = '/'.join(original_dataset)
	
	# Load the splits
	train_set, val_set, test_set = create_splits(
		original_dataset,
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

	# Select the classifier
	classifier = classifiers[classifier_name]
	clf_name = classifier_name

	# For svc_linear use only a random subset of the dataset to train
	if 'svc' in classifier_name and len(y_train) > 200000:
		rand_ind = np.random.randint(low=0, high=len(y_train), size=200000)
		X_train = X_train.iloc[rand_ind]
		y_train = y_train.iloc[rand_ind]

	# Fit the classifier
	print(f'----------------------------------\nTraining {names[classifier_name]}...')
	tic = perf_counter()
	classifier.fit(X_train, y_train)
	toc = perf_counter()

	# Print training time
	training_stats["training_time"] = toc - tic
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
	classifier_name = f"{clf_name}_{window_size}"
	timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
	df = pd.DataFrame.from_dict(training_stats, columns=["training_stats"], orient="index")
	df.to_csv(os.path.join(save_done_training, f"{classifier_name}_{timestamp}.csv"))

	# Save pipeline
	saving_dir = os.path.join(path_save, classifier_name) if classifier_name.lower() not in path_save.lower() else path_save
	saved_model_path = save_classifier(classifier, saving_dir, fname=None)

	# Evaluate on test set or val set
	if eval_model:
		eval_set = test_indexes if len(test_indexes) > 0 else val_indexes
		eval_feature_based(
			data_path=data_path, 
			model_name=classifier_name,
			model_path=saved_model_path,
			path_save=path_save_results,
			fnames=eval_set,
		)


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
	parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')
	parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="results/weights")

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
			eval_model=args.eval_true,
			path_save=args.path_save
		)
