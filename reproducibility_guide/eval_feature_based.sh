#!/bin/bash

# Nearest Neighbors
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=knn --model_path=results/weights/supervised/knn_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=knn --model_path=results/weights/supervised/knn_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=knn --model_path=results/weights/supervised/knn_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=knn --model_path=results/weights/supervised/knn_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=knn --model_path=results/weights/supervised/knn_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=knn --model_path=results/weights/supervised/knn_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=knn --model_path=results/weights/supervised/knn_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=knn --model_path=results/weights/supervised/knn_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Linear SVM
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=svc_linear --model_path=results/weights/supervised/svc_linear_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Decision Tree
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=decision_tree --model_path=results/weights/supervised/decision_tree_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Random Forest
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=random_forest --model_path=results/weights/supervised/random_forest_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=random_forest --model_path=results/weights/supervised/random_forest_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=random_forest --model_path=results/weights/supervised/random_forest_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=random_forest --model_path=results/weights/supervised/random_forest_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=random_forest --model_path=results/weights/supervised/random_forest_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=random_forest --model_path=results/weights/supervised/random_forest_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=random_forest --model_path=results/weights/supervised/random_forest_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=random_forest --model_path=results/weights/supervised/random_forest_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Neural Net
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=mlp --model_path=results/weights/supervised/mlp_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=mlp --model_path=results/weights/supervised/mlp_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=mlp --model_path=results/weights/supervised/mlp_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=mlp --model_path=results/weights/supervised/mlp_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=mlp --model_path=results/weights/supervised/mlp_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=mlp --model_path=results/weights/supervised/mlp_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=mlp --model_path=results/weights/supervised/mlp_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=mlp --model_path=results/weights/supervised/mlp_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv

# AdaBoost
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=ada_boost --model_path=results/weights/supervised/ada_boost_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Naive Bayes
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=bayes --model_path=results/weights/supervised/bayes_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=bayes --model_path=results/weights/supervised/bayes_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=bayes --model_path=results/weights/supervised/bayes_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=bayes --model_path=results/weights/supervised/bayes_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=bayes --model_path=results/weights/supervised/bayes_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=bayes --model_path=results/weights/supervised/bayes_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=bayes --model_path=results/weights/supervised/bayes_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=bayes --model_path=results/weights/supervised/bayes_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv

# QDA
python3 eval_feature_based.py --data=data/TSB_16/TSFRESH_TSB_16.csv --model=qda --model_path=results/weights/supervised/qda_16/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_feature_based.py --data=data/TSB_32/TSFRESH_TSB_32.csv --model=qda --model_path=results/weights/supervised/qda_32/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_feature_based.py --data=data/TSB_64/TSFRESH_TSB_64.csv --model=qda --model_path=results/weights/supervised/qda_64/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_feature_based.py --data=data/TSB_128/TSFRESH_TSB_128.csv --model=qda --model_path=results/weights/supervised/qda_128/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_feature_based.py --data=data/TSB_256/TSFRESH_TSB_256.csv --model=qda --model_path=results/weights/supervised/qda_256/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=qda --model_path=results/weights/supervised/qda_512/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_feature_based.py --data=data/TSB_768/TSFRESH_TSB_768.csv --model=qda --model_path=results/weights/supervised/qda_768/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_feature_based.py --data=data/TSB_1024/TSFRESH_TSB_1024.csv --model=qda --model_path=results/weights/supervised/qda_1024/ --path_save=results/raw_predictions/supervised/ --file=experiments/supervised_splits/split_TSB_1024.csv
