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




def eval_rocket(data_path, model_path, read_from_file=None, data=None):
	"""Predict some time series with the given rocket model

	:param data_path:
	:param model_name:
	:param read_from_file:
	:param data:
	"""

	# Load model (rocket BUT maybe also any other feature based clf)

	# Load data (do this for deep learning too - the thing with the data and read_from_file)
	if data is None:
		if read_from_file is None:
			pass # Load every time series under this dir similar to eval_deep_model
		else:
			pass # Find test (if no test, then val), and load every timeseries name in there

	# Call the evaluator

	# Compute the predictions and the inference time per time series (let use choose for inf)
		# Add inf time for deep learning too

	# Print results

	# Save the results (Somewhere in results but don't know where in specific)
		# Save results for deep learning too







'''
	# Save pipeline
	saving_dir = os.path.join(path_save, classifier_name) if classifier_name.lower() not in path_save.lower() else path_save
	save_classifier(classifier, saving_dir, fname=None)
	classifier = load_classifier(saving_dir)

	# Print training information and accuracy on validation set
	toc = perf_counter()
	print("training time: {:.3f} secs".format(toc-tic))
	tic = perf_counter()
	classifier_score = classifier.score(X_val, y_val)
	toc = perf_counter()
	print('valid accuracy: {:.3%}'.format(classifier_score))
	print("inference time: {:.3} ms".format(((toc-tic)/X_val.shape[0]) * 1000))
	exit()

	# Load scores
	# test_set = test_set[:5]
	data_loader = DataLoader(TSB_data_path)
	eval_set = test_set if unsupervised else val_set
	fnames = [x[:-4] for x in eval_set]
	
	# Load metrics
	metricsloader = MetricsLoader(TSB_new_metrics_path)

	# Evaluating the model
	evaluator = Evaluator()
	for metric in metrics:
			metrics = metricsloader.read(metric=metric).loc[fnames]
			curr_metrics = evaluator.compute_anom_score_simple(
					model=classifier,
					model_type="rocket",
					fnames=eval_set,
					metric_values=metrics[detector_names],
					metric=metric,
					data_path=data_path,
					batch_size=batch_size
			)
			curr_metrics.columns = ["rocket_{}_{}".format(str(window_size), x) for x in curr_metrics.columns.values]

			# Save scores
			model_name = '_'.join([classifier_name, str(seed), str(keep_labels), str(window_size)])
			if unsupervised:
					unsupervised_name = str(unsupervised).split('/')[-1][:-4].replace('unsupervised_', '')
					file_name = os.path.join("unsupervised_model_scores", metric, "{}_{}_{}.csv".format(model_name, metric, unsupervised_name))
			else:
					file_name = os.path.join("model_scores", metric, "{}_{}.csv".format(model_name, metric))
			curr_metrics.to_csv(file_name)
'''