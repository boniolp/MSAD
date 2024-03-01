#!/bin/bash

python3 train_rocket.py --path=data/TSB_16/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/TSB_32/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/TSB_64/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/TSB_128/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/TSB_256/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/TSB_512/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/TSB_768/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --eval-true --path_save=results/weights/
python3 train_rocket.py --path=data/TSB_1024/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --eval-true --path_save=results/weights/