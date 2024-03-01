# Reproducability Guide

## Prepare the dataset

First you will have to download the dataset according to the first guide. In there we added a new folder that contains the dataset preprocessed with all the additional data that you will need to run our experiments. We did this because to produce the featured dataset requires lots of time, memory and processing power (we used a server with 512 GB of RAM and 32 cores). Thus, you can skip the first step of preparing the dataset if you download the completed dataset. Find the links in the readme file in the data folder.

To create the additional data on your own, without downloading them from our drive, run the following command
```
bash data_preparation.sh
```
To do this you will have to download the TSB.zip folder from our drive. The data.zip folder contains everything so you can skip this step.


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

Once you have the results from all the models in the results/raw_predictions directory, run the following command to combine them into a single file:

```
python3 merge_scores.py --path=results/raw_predictions/ --metric=auc_pr --save_path=results/
python3 merge_scores.py --path=results/raw_predictions/ --metric=vus_pr --save_path=results/
```

Now you can run the notebook with the following command:
```
jupyter notebook
```
Follow the comments in the notebook and run every cell to see the results. 

NOTE: For the time being the notebook requires the results from all models to run otherwise it crashes. This is incovenient as you may only want to see some of the results. We are planning to change the notebook so that it can run with only a part of the results.

NOTE: This guide does not yet contain information on how to reproduce the inference results and the unsupervised experiments. We will add those next week.

For any questions contact me on my email:
emmanouil.sylligardos@ens.fr

Thank you for your time and I would be happy to answer any of your questions regarding reproducing our experiments.
