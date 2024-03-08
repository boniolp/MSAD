# Reproducibility Guide

## Prepare the dataset

First, you will have to download the dataset according to the instructions in the first guide. In there, we added a new folder that contains the preprocessed dataset with all the additional data required to run our experiments. We did this because producing the featured dataset requires a significant amount of time, memory, and processing power (we used a server with 512 GB of RAM and 32 cores). Thus, you can skip the first step of preparing the dataset if you download the completed dataset. Find the links in the README file in the `data` folder.

To create the additional data on your own, without downloading it from our drive, run the following command:

```
bash data_preparation.sh
```

To do this, you will have to download the `TSB.zip` folder from our drive. The `data.zip` folder contains everything, so you can skip this step. Here is the link for the preprocessed/ready-to-go dataset:
https://drive.google.com/file/d/1KBFzKE3Z-tUe_3KdI6gxnfjbMc0ampr6/view?usp=sharing

## Train all models and evaluate them on the validation data

We have created multiple `.sh` files to reproduce the results. Some of them are for training the models, and some are for evaluating them. Training all the models requires a significant amount of computing power and time. Thus, you can find all the trained models in the `weights.zip` folder on our drive (link provided in the README files in the `results/weights` directory), which includes deep learning, feature-based, and rocket models (the last two were added recently).

To train all the models again, run the following commands from the root directory:

```
bash train_deep_models.sh
bash train_feature_based.sh
bash train_rockets.sh
```

All these commands already contain a command-line argument for the Python scripts to train the models, save the trained models in the `results/weights` directory, and evaluate the trained models on the validation data, saving the results in the `results/raw_predictions` directory. So, if you want to skip the step of training the models and use our already trained models for evaluation, proceed to the next step. However, if you want to train the models, then the evaluation results will be produced automatically, so you can skip the next step.

## Evaluating our trained models to reproduce the results

You can use our trained models to create files with their predictions on the validation dataset. To do this, run the following commands:

```
bash eval_deep_models.sh
bash eval_feature_based.sh
bash eval_rockets.sh
```

This command will use our trained models (or yours if you performed the previous step) to evaluate them on the validation set. For each model (128 in total), a file will be saved in the `results/raw_predictions` directory with the predictions for each time series in the validation set.

## Combine the results and run the jupyter notebook

Once you have the results from _all_ the models in the `results/raw_predictions` directory, run the following commands to combine them into a single file:

```
python3 merge_scores.py --path=results/raw_predictions/ --metric=auc_pr --save_path=results/
python3 merge_scores.py --path=results/raw_predictions/ --metric=vus_pr --save_path=results/
```

Now you can run the Jupyter Notebook with the following command:

```
jupyter notebook
```

Follow the comments in the notebook and run every cell to see the results.

## Inference and Execution Time

### Inference Time

In the `results/execution_time` directory, you will find all files related to execution time. Some specifications:
- Inference time for the detectors is the time required for one detector (e.g., NORMA) to process one time series.
- Inference time for a model selector is the time required to predict the best detector for a given time series + the inference time of that detector.
- Training time is the time required to train all model selectors on each dataset.

In the `all_inference_time.csv` file, there are the combined inference times for the detectors + Averaging Ensemble + model selectors (according to their predictions at the time of publishing the paper), and we use this file to produce the plots in the notebooks. Below, we provide a method to validate our results with your results:

- A `detectors_inference_time.csv` file was created that contains separately the inference time of only the detectors and the Averaging Ensemble. We will use this file to compute the inference time of our model selectors.
- Once you have trained (or used our pretrained weights) and evaluated the trained models on the validation data, you should have a file with the predictions of the model in the `results/raw_predictions/` directory. There is a separate file for each model.
- This file contains the prediction of the model for every time series + the inference time, which is ONLY the time to make the prediction. We also refer to this value as "Prediction time."
- Run the following command to compute inference time results, and then run the notebooks to validate the results:

```
python3 merge_scores.py --path=results/raw_predictions/ --save_path=results/ --time-true
```

In the `all_training_time.csv` file, there are the training times for every model selector and dataset, with some additional information on the datasets and their processing times. To validate these results on your own, please do the following:

- Train a model selector, let's say ConvNet 512, which is the ConvNet model on the dataset of windows with a length of 512 points.
- After training is completed, every model selector will save a file in the `results/done_training/` directory with information regarding the training process. Check the corresponding file for the model selector you trained.
- In that file, you will find the training time. Manually validate it with the times in the `all_training_time.csv` file. Please refer to the paper for information on the training process.

In the `detectors_inference_time.csv` file, you will find the inference time for all detectors and for every time series in the validation set + the inference time of the Averaging Ensemble (Avg Ens.), which is the sum of the inference time of all detectors in a given time series.

## Unsupervised Experiments

For the unsupervised experiments, we are only using 5 models, which are our best models (at the time of publication) per family of models. Namely, the models are kNN-1024, Rocket-128, ConvNet-128, ResNet-1024, and SiT-512. We are training each model 16 different times on 16 different subsets of the TSB benchmark, each time leaving one of the 16 datasets out of the training process for testing. Then, we are evaluating the models on the one dataset that was left out.

### Train the models for the unsupervised experiments

To train the models, we provide three `.sh` files. If you don't want to train the models, since it is very time-consuming and requires a lot of processing power, you can use our trained models that you will find in the `weights` directory on the cloud (link provided in the main README of the repository). However, the unsupervised deep learning models are not provided, so you can either evaluate our results on 2 out of the 5 models or train them yourself using one of the following commands. The commands to run the `.sh` files and train the unsupervised models:

```
bash train_unsupervised_deep_models.sh
bash train_unsupervised_feature_based.sh
bash train_unsupervised_rockets.sh
```

These commands will train the models on all different splits/subsets of the dataset and will save the trained models. Rocket and kNN (a.k.a. feature-based) will be saved in `results/weights/unsupervised/`, but deep models will be saved in `results/weights/`. Make sure to manually move the deep models into the `results/weights/unsupervised/` directory.

### Evaluating the trained models

Whether you downloaded our trained models or trained them yourself, you should now have the trained weights in the `results/weights/unsupervised/` directory. You can now run the `.sh` files to evaluate the models, that is, to predict for each model the one dataset that was left out from the training process and has never been seen by the model. The commands to evaluate the models:

```
bash eval_unsupervised_deep_models.sh
bash eval_unsupervised_feature_based.sh
bash eval_unsupervised_rockets.sh
```

These commands will save the predictions of the models for each split in the `results/raw_predictions/unsupervised/` directory.

### Combine the unsupervised results and visualize them

If you weren't able to train the deep models, you are able to visualize part of the results by commenting and uncommenting certain parts of the notebook. First, we have to merge our results. To do this, run the following command:

```
python3 merge_unsupervised_scores.py --path=results/raw_predictions/unsupervised/ --metric=auc_pr --save_path=results/unsupervised_results/AUC_PR/
```

This command will merge the results of all models, per split, so after running this command, you should have a file per split of the dataset, and not a file per split per model. The combined results will be saved in the `results/unsupervised_results/AUC_PR/` directory under the names `current_testsize_1_split_#.csv`. This is to differentiate to our original results from the publication. You can modify the reading path on the notebook to either display the plots from your reproduced results and our original results.

Note: Although we provide most of our models trained so you can evaluate them directly without training them, we do not provide yet the unsupervised deep models trained. To evaluate those you have to train them, otherwise you can evaluate the unsupervised experiments with just the feature-based and rocket models which we do provide trained.

For any questions, you can contact me by email at:
emmanouil.sylligardos@ens.fr

Thank you for your time to reproduce our results, and I would be happy to answer any of your questions.
