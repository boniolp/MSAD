# Reproducibility Guide

## Prepare the dataset

First you will have to download the dataset according to the first guide. In there we added a new folder that contains the dataset preprocessed with all the additional data that you will need to run our experiments. We did this because to produce the featured dataset requires lots of time, memory and processing power (we used a server with 512 GB of RAM and 32 cores). Thus, you can skip the first step of preparing the dataset if you download the completed dataset. Find the links in the readme file in the data folder.

To create the additional data on your own, without downloading them from our drive, run the following command

```
bash data_preparation.sh
```

To do this you will have to download the TSB.zip folder from our drive. The data.zip folder contains everything so you can skip this step. Here is the link for the preprocessed/ready-to-go dataset:
https://drive.google.com/file/d/1KBFzKE3Z-tUe_3KdI6gxnfjbMc0ampr6/view?usp=sharing

## Train all models and evaluate them on the validation data

We have created multiple .sh files to reproduce the results. Some of them are to train the models and some of them are to evaluate them. Training all the models requires again lots of computing power and time, thus you can find all the trained models into the weights.zip folder in our drive (link in the readme files in the results/weights directory), that is deep learning, feature based and rockets (last two added lately).

For training all the models again run the following commands from the root directory.

```
bash train_deep_models.sh
bash train_feature_based.sh
bash train_rockets.sh
```

All these commands already contain an command line argument for the python scripts to train the models, save the trained model in the results/weights directory and evaluate the trained model on the validation data and save the results into the results/raw_predictions directory. So, if you want to skip the step of training the models and you want to use our already trained models to evaluate them you should do the next step. But if you want to train them, you can skip the next step

## Evaluating our trained models to reproduce the results

You can use our trained models to create the files with their predictions on the validation dataset. To do it run the following commands:

```
bash eval_deep_models.sh
bash eval_feature_based.sh
bash eval_rockets.sh
```

This command will use our trained models (or yours if you did the previous step) to evaluate them on the validation set. For each model (128 in total) a file will be saved in the results/raw_predictions directory with the predictions for each time series in the validation step.

## Combine the results and run the jupyter notebook

Once you have the results from _all_ the models in the results/raw_predictions directory, run the following command to combine them into a single file:

```
python3 merge_scores.py --path=results/raw_predictions/ --metric=auc_pr --save_path=results/
python3 merge_scores.py --path=results/raw_predictions/ --metric=vus_pr --save_path=results/
```

Now you can run the notebook with the following command:

```
jupyter notebook
```

Follow the comments in the notebook and run every cell to see the results.

## Inference and Execution Time

### Inference Time

In the directory 'results/execution_time' you will find all files having to do with time. Some specifications: 
- Inference time for the detectors is the time one detector (e.g. NORMA) requires to digest one time series.
- Inference time for a model selector is the time required to predict the best detector for a given time series + the inference time of that detector.
- Training time is the time required to train all model selectors on each dataset.

In the 'all_inference_time.csv' there are the combined inference times for the detectors + Averaging Ensemble + model selectors (according to their predictions at the time of publishing the paper) and we use this file to produce the plots on the notebooks. Below we provide a method to validate our results with your results:
- A 'detectors_inference_time.csv' file was created that containts seperately the inference time of only the detectors and the averaging ensemble. We will use this file to compute the inference time of our model selectors.
- Once you have trained (or used our pretrained weights) and have evaluated the trained models on the validation data, you should have a file of the predictions of the model in the 'results/raw_predictions/' directory. There is a seperate file for each model
- This file contains the prediction of the model for every time series + the inference time, that is ONLY the time to make the prediction. We also refer to this value as "Prediction time"
- Run the following command to compute inference time results and then run the notebooks to validate the results
```
python3 merge_scores.py --path=results/raw_predictions/ --save_path=results/ --time-true
```

In the 'all_training_time.csv' there are the training times for every model selector and dataset, with some additional information on the dataset and their processing times. To validate these results on your own please do the following:
- Train some model selector, let's say ConvNet 512, that is the ConvNet model on the dataset of windows with length 512 points.
- After training is completed, every model selector will save a file in the 'results/done_training/' directory with information regarding the training process. Check the corresponding file for the model selector you trained.
- In that file you will find the training time, manually validate it with the times in the 'all_training_time.csv' file. Please refer to the paper for information on the training process.

In the 'detectors_inference_time.csv' file you will find the inference time for all detectors and for every time series in the validation set + the inderence time of Averaging Ensemble (Avg Ens.) which is the sum of the inference time of all detectors in a given time series.

```

```


NOTE: For the time being the notebook requires the results from all models to run otherwise it crashes. This is incovenient as you may only want to see some of the results. We are planning to change the notebook so that it can run with only a part of the results.

NOTE: This guide does not yet contain information on how to reproduce the inference results and the unsupervised experiments. We will add those next week.

For any questions contact me on my email:
emmanouil.sylligardos@ens.fr

Thank you for your time and I would be happy to answer any of your questions regarding reproducing our experiments.
