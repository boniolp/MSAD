#!/bin/bash

# Evaluate convnet
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_0 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_0.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_1 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_1.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_2 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_2.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_3 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_3.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_4 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_4.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_5 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_5.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_6 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_6.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_7 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_7.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_8 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_8.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_9 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_9.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_10 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_10.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_11 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_11.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_12 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_12.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_13 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_13.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_14 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_14.csv
python3 eval_deep_model.py --data=data/TSB_128/ --model=convnet --model_path=results/weights/unsupervised/convnet_default_128_testsize_1_split_15 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_15.csv

# Evaluate Resnet
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_0 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_0.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_1 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_1.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_2 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_2.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_3 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_3.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_4 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_4.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_5 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_5.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_6 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_6.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_7 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_7.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_8 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_8.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_9 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_9.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_10 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_10.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_11 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_11.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_12 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_12.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_13 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_13.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_14 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_14.csv
python3 eval_deep_model.py --data=data/TSB_1024/ --model=resnet --model_path=results/weights/unsupervised/resnet_default_1024_testsize_1_split_15 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_15.csv

# Evaluate Signal Transformer (SiT) with Original Stem
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_0 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_0.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_1 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_1.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_2 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_2.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_3 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_3.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_4 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_4.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_5 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_5.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_6 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_6.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_7 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_7.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_8 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_8.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_9 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_9.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_10 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_10.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_11 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_11.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_12 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_12.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_13 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_13.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_14 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_14.csv
python3 eval_deep_model.py --data=data/TSB_512/ --model=sit --model_path=results/weights/unsupervised/sit_stem_original_512_testsize_1_split_15 --params=models/configuration/sit_stem_original.json --path_save=results/raw_predictions/unsupervised/ --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_15.csv
