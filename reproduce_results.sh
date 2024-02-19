#!/bin/bash

# Evaluate all pretrained deep learning models
python3 eval_deep_model.py --data=data/TSB_16/ --model=convnet --model_path=results/weights/supervised/convnet_default_16 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv