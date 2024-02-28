#!/bin/bash

# Evaluate convnet
# python3.6 eval_deep_model.py --data=data/TSB_16/ --model=convnet --model_path=results/weights/supervised/convnet_default_16 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
# python3.6 eval_deep_model.py --data=data/TSB_32/ --model=convnet --model_path=results/weights/supervised/convnet_default_32 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
# python3.6 eval_deep_model.py --data=data/TSB_64/ --model=convnet --model_path=results/weights/supervised/convnet_default_64 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
# python3.6 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/supervised/convnet_default_128 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3.6 eval_deep_model.py --data=data/TSB_256/ --model=convnet --model_path=results/weights/supervised/convnet_default_256 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3.6 eval_deep_model.py --data=data/TSB_512/ --model=convnet --model_path=results/weights/supervised/convnet_default_512 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3.6 eval_deep_model.py --data=data/TSB_768/ --model=convnet --model_path=results/weights/supervised/convnet_default_768 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3.6 eval_deep_model.py --data=data/TSB_1024/ --model=convnet --model_path=results/weights/supervised/convnet_default_1024 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Evaluate Inception Time
python3.6 eval_deep_model.py --data=data/TSB_16/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_16 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
python3.6 eval_deep_model.py --data=data/TSB_32/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_32 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
python3.6 eval_deep_model.py --data=data/TSB_64/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_64 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
python3.6 eval_deep_model.py --data=data/TSB_128/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_128 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3.6 eval_deep_model.py --data=data/TSB_256/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_256 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3.6 eval_deep_model.py --data=data/TSB_512/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_512 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3.6 eval_deep_model.py --data=data/TSB_768/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_768 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3.6 eval_deep_model.py --data=data/TSB_1024/ --model=inception_time --model_path=results/weights/supervised/inception_time_default_1024 --params=models/configuration/inception_time_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Evaluate Resnet
python3.6 eval_deep_model.py --data=data/TSB_16/ --model=resnet --model_path=results/weights/supervised/resnet_default_16 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
python3.6 eval_deep_model.py --data=data/TSB_32/ --model=resnet --model_path=results/weights/supervised/resnet_default_32 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
python3.6 eval_deep_model.py --data=data/TSB_64/ --model=resnet --model_path=results/weights/supervised/resnet_default_64 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
python3.6 eval_deep_model.py --data=data/TSB_128/ --model=resnet --model_path=results/weights/supervised/resnet_default_128 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3.6 eval_deep_model.py --data=data/TSB_256/ --model=resnet --model_path=results/weights/supervised/resnet_default_256 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3.6 eval_deep_model.py --data=data/TSB_512/ --model=resnet --model_path=results/weights/supervised/resnet_default_512 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3.6 eval_deep_model.py --data=data/TSB_768/ --model=resnet --model_path=results/weights/supervised/resnet_default_768 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3.6 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/supervised/resnet_default_1024 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Evaluate Signal Transformer (SiT) with Convolutional Patch
python3.6 eval_deep_model.py --data=data/TSB_16/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_16 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
python3.6 eval_deep_model.py --data=data/TSB_32/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_32 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
python3.6 eval_deep_model.py --data=data/TSB_64/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_64 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
python3.6 eval_deep_model.py --data=data/TSB_128/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_128 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3.6 eval_deep_model.py --data=data/TSB_256/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_256 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3.6 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_512 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3.6 eval_deep_model.py --data=data/TSB_768/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_768 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3.6 eval_deep_model.py --data=data/TSB_1024/ --model=sit --model_path=results/weights/supervised/sit_conv_patch_1024 --params=models/configuration/sit_conv_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Evaluate Signal Transformer (SiT) with Linear Patch
python3.6 eval_deep_model.py --data=data/TSB_16/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_16 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
python3.6 eval_deep_model.py --data=data/TSB_32/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_32 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
python3.6 eval_deep_model.py --data=data/TSB_64/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_64 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
python3.6 eval_deep_model.py --data=data/TSB_128/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_128 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3.6 eval_deep_model.py --data=data/TSB_256/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_256 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3.6 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_512 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3.6 eval_deep_model.py --data=data/TSB_768/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_768 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3.6 eval_deep_model.py --data=data/TSB_1024/ --model=sit --model_path=results/weights/supervised/sit_linear_patch_1024 --params=models/configuration/sit_linear_patch.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Evaluate Signal Transformer (SiT) with Original Stem
python3.6 eval_deep_model.py --data=data/TSB_16/ --model=sit --model_path=results/weights/supervised/sit_stem_original_16 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
python3.6 eval_deep_model.py --data=data/TSB_32/ --model=sit --model_path=results/weights/supervised/sit_stem_original_32 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
python3.6 eval_deep_model.py --data=data/TSB_64/ --model=sit --model_path=results/weights/supervised/sit_stem_original_64 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
python3.6 eval_deep_model.py --data=data/TSB_128/ --model=sit --model_path=results/weights/supervised/sit_stem_original_128 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3.6 eval_deep_model.py --data=data/TSB_256/ --model=sit --model_path=results/weights/supervised/sit_stem_original_256 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3.6 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/supervised/sit_stem_original_512 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3.6 eval_deep_model.py --data=data/TSB_768/ --model=sit --model_path=results/weights/supervised/sit_stem_original_768 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3.6 eval_deep_model.py --data=data/TSB_1024/ --model=sit --model_path=results/weights/supervised/sit_stem_original_1024 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv

# Evaluate Signal Transformer (SiT) with ReLU Stem
python3.6 eval_deep_model.py --data=data/TSB_16/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_16 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_16.csv
python3.6 eval_deep_model.py --data=data/TSB_32/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_32 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_32.csv
python3.6 eval_deep_model.py --data=data/TSB_64/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_64 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_64.csv
python3.6 eval_deep_model.py --data=data/TSB_128/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_128 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
python3.6 eval_deep_model.py --data=data/TSB_256/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_256 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_256.csv
python3.6 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_512 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
python3.6 eval_deep_model.py --data=data/TSB_768/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_768 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_768.csv
python3.6 eval_deep_model.py --data=data/TSB_1024/ --model=sit --model_path=results/weights/supervised/sit_stem_relu_1024 --params=models/configuration/sit_stem_relu.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_1024.csv