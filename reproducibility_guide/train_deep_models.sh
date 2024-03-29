#!/bin/bash

# Training Convnet
python3 train_deep_model.py --path=data/TSB_16/ --split=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_32/ --split=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_64/ --split=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_256/ --split=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_768/ --split=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=10 --eval-true

# Training Inception Time
python3 train_deep_model.py --path=data/TSB_16/ --split=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_32/ --split=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_64/ --split=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_256/ --split=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_768/ --split=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --model=inception_time --params=models/configuration/inception_time_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --model=inception_time --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true

# Training Resnet
python3 train_deep_model.py --path=data/TSB_16/ --split=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_32/ --split=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_64/ --split=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_256/ --split=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_768/ --split=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true


# Training SiT with Convolutional Patch
python3 train_deep_model.py --path=data/TSB_16/ --split=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_32/ --split=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_64/ --split=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_256/ --split=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_768/ --split=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --model=sit --params=models/configuration/sit_conv_patch.json --batch=256 --epochs=10 --eval-true


# Training SiT with Linear Patch
python3 train_deep_model.py --path=data/TSB_16/ --split=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_32/ --split=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_64/ --split=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_256/ --split=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_768/ --split=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --model=sit --params=models/configuration/sit_linear_patch.json --batch=256 --epochs=10 --eval-true


# Training SiT with Original Stem
python3 train_deep_model.py --path=data/TSB_16/ --split=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_32/ --split=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_64/ --split=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_256/ --split=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_768/ --split=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=10 --eval-true


# Training SiT with ReLU Stem
python3 train_deep_model.py --path=data/TSB_16/ --split=0.7 --file=experiments/supervised_splits/split_TSB_16.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_32/ --split=0.7 --file=experiments/supervised_splits/split_TSB_32.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_64/ --split=0.7 --file=experiments/supervised_splits/split_TSB_64.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_256/ --split=0.7 --file=experiments/supervised_splits/split_TSB_256.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_768/ --split=0.7 --file=experiments/supervised_splits/split_TSB_768.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/supervised_splits/split_TSB_1024.csv --model=sit --params=models/configuration/sit_stem_relu.json --batch=256 --epochs=10 --eval-true
