#!/bin/bash

# Training ConvNet-128
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_0.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_1.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_2.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_3.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_4.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_5.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_6.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_7.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_8.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_9.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_10.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_11.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_12.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_13.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_14.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_15.csv --model=convnet --params=models/configuration/convnet_default.json --batch=256 --epochs=100

# Training ResNet-1024
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_0.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_1.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_2.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_3.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_4.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_5.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_6.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_7.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_8.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_9.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_10.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_11.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_12.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_13.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_14.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_1024/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_15.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=100

# Training SiT-512
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_0.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_1.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_2.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_3.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_4.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_5.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_6.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_7.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_8.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_9.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_10.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_11.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_12.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_13.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_14.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_15.csv --model=sit --params=models/configuration/sit_stem_original.json --batch=256 --epochs=100


