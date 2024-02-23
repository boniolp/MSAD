#!/bin/bash

python3 eval_rocket.py --data=data/TSB_16/ --model_path=results/weights/supervised/rocket_16/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
python3 eval_rocket.py --data=data/TSB_32/ --model_path=results/weights/supervised/rocket_32/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
python3 eval_rocket.py --data=data/TSB_64/ --model_path=results/weights/supervised/rocket_64/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
python3 eval_rocket.py --data=data/TSB_128/ --model_path=results/weights/supervised/rocket_128/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3 eval_rocket.py --data=data/TSB_256/ --model_path=results/weights/supervised/rocket_256/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3 eval_rocket.py --data=data/TSB_512/ --model_path=results/weights/supervised/rocket_512/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3 eval_rocket.py --data=data/TSB_768/ --model_path=results/weights/supervised/rocket_768/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3 eval_rocket.py --data=data/TSB_1024/ --model_path=results/weights/supervised/rocket_1024/ --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv
