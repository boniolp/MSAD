<p align="center">
<img width="150" src="./assets/figures/MSAD.png"/>
</p>


<h1 align="center">MSAD</h1>
<h2 align="center">Model Selection for Anomaly Detection in Time Series</h2>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/boniolp/MSAD"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/boniolp/MSAD">
</p>
</div>


<p align="center"><a href="https://adecimots.streamlit.app/">Try our demo</a></p>

MSAD proposes a pipeline for model selection based on time series classification and an extensive experimental evaluation of existing classification algorithms for this new pipeline. Our results demonstrate that model selection methods outperform every single anomaly detection method while being in the same order of magnitude regarding execution time. You can click on our demo link above to get more information and navigate through our experimental evaluation.


## Contributors

* Emmanouil Sylligardos (ICS-FORTH)
* Paul Boniol (Université Paris Cité)

## Installation

The following tools are required to install MSAD from source:

- git
- conda (anaconda or miniconda)

#### Steps

1. First, due to limitations in the upload size on GitHub, we host the datasets and pretrained models at a different location. Please download the datasets using the following links:

- datasets: https://drive.google.com/file/d/1PQKwu5lZTnTmsmms1ipko9KF712Oc5DE/view?usp=share_link

- models: https://drive.google.com/file/d/14Tk_-npHIozLAYB-FAkHRtTn9CqaXwuE/view?usp=share_link

Unzip the files and move the datasets (i.e., TSB/ folder) in data/, and move the models files (i.e., the contents of the unzipped file called weight) in weights/ folder in the repo.

2. Clone this repository using git and change into its root directory.

```bash
git clone https://github.com/boniolp/MSAD.git
cd MSAD/
```

3. Create and activate a conda-environment 'MSAD'.

```bash
conda env create --file environment.yml
conda activate MSAD
```
   
4. Install the dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```

## Usage

#### Compute Oracle
The Oracle (in white in the results figure at the end) is a hypothetical model that simulates the accuracy of a model on a given benchmark and evaluates its anomaly detection ability. Oracle can be simulated with different accuracy values, ranging from 1 (always selects the best detector for a time series) to zero (always selects a wrong detector). Additonally, Oracle can simulate different modes of randomness, namely:
1) true - when wrong, randomly select another detector,
2) lucky - when wrong, always select the second best detector (upper bound),
3) unlucky - when wrong, always select the worst detector (lower bound),
4) best-k - when wrong, always select the kth detector (e.g. best-2 is lucky)

To compute Oracle run the following command:

```bash
python3 run_oracle.py --path=data/TSB/metrics/ --acc=1 --randomnes=true
```
- path: path to metrics (the results will be saved here)
- acc: the accuracy that you want to simulate (float between 0 and 1)
- randomness: the randomness that you want to simulate (see possible modes above)

#### Compute Average Ensembling
The Average Ensembling, or Avg Ens (in orange in the results figure at the end) is to ensemble the anomaly scores produced by all the detectors, that is, to compute their average. Then, the AUC-PR and the VUS-PR metrics are computed for the resulted score.

To compute Avg Ens run the following command:
```bash
python3 run_avg_ens.py --n_jobs=16
```
- n_jobs: threads to use for parallel computation

#### Data Preprocessing
Our models have been implemented so that the input is of a fixed size. Thus, before we run any models, we first devide every dataset in the TSB benchmark into windows. Note that you can add your own time series here and divide them into windows, just make sure to follow the same format.

To produce a windowed dataset, run the following command:
```bash
python3 create_windows_dataset.py --save_dir=data/ --path=data/TSB/data/ --metric_path=data/TSB/metrics/ --window_size=512 --metric=AUC_PR
```
- save_dir: path to save the dataset
- path: path of the dataset to divide into windows
- metric_path: path to the metrics of the dataset given (to produce the labels)
- window_size: window size (if window size bigger than the time series' length then this time series is skipped)
- metric: metric to use to produce the labels (AUC-PR, VUS-PR, AUC-ROC, VUS-ROC)

The feature-based methods, require a set of features to be computed first and turn the time series into tabular data. To this goal we use the TSFresh module that computes a predifined set of features.

To compute the set of features for a segmented dataset run the following command:
```bash
python3 generate_features.py --path=data/TSB_512/
```
--path: path to the dataset to compute the features (the dataset should be segmented first into windows - see the command above), the resulting dataset is saved in the same dir

#### Deep Learning Architectures

To train a model, run the following command:
```bash
python3 train_deep_model.py --path=data/TSB_512/ --split=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --model=resnet --params=models/configuration/resnet_default.json --batch=256 --epochs=10 --eval-true
```
- path: path to the dataset to use
- split: split percentage for train and val sets
- seed: Seed for train/val split
- file: path to file that contains a specific split (to reproduce our results)
- model: model to run (type of architecture)
- params: a json file with the model's parameters
- batch: batch size
- epochs: number of epochs
- eval-true: whether to evaluate the model on test data after training

To evaluate a model on a folder of csv files, run the following command:
```bash
python3 eval_deep_model.py --data=data/TSB_512/MGAB/ --model=convnet --model_path=results/weights/supervised/convnet_default_512/model_30012023_173428 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/
```
- data: path to the time series to predict
- model: model to run
- model_path: path to the trained model
- params: a json file with the model's parameters
- path_save: path to save the results

To reproduce our results, run the following command:
```bash
python3 eval_deep_model.py --data=data/TSB_512/ --model=convnet --model_path=results/weights/supervised/convnet_default_512/model_30012023_173428 --params=models/configuration/convnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_512.csv
```
- file: path to file that contains a specific split (to reproduce our results)

#### Rocket

To train a Rocket model, run the following command:
```bash
python3 train_rocket.py --path=data/TSB_512/ --split_per=0.7 --file=experiments/supervised_splits/split_TSB_512.csv --eval-true --path_save=results/weights/supervised/
```
- path: path to the dataset to use
- split_per: split percentage for train and val sets
- seed: seed for splitting train, val sets (use small number)
- file: path to file that contains a specific split
- eval-true: whether to evaluate the model on test data after training
- path_save: path to save the trained classifier

To evaluate a Rocket model, run the following command:
```bash
python3 eval_rocket.py --data=data/TSB_512/KDD21/ --model_path=results/weights/supervised/rocket_512/ --path_save=results/raw_predictions/
```
- data: path to the time series to predict
- model_path: path to the trained model
- path_save: path where to save the results

#### Feature-Based
list of classifiers:
- knn
- svc_linear
- decision_tree
- random_forest
- mlp
- ada_boost
- bayes
- qda

To train any of these classifiers, run the following command:
```bash
python3 train_feature_based.py --path=data/TSFRESH_TSB_512.csv --classifier=knn --split_per=0.7 --file=experiments/unsupervised_splits/unsupervised_testsize_1_split_0.csv --eval-true --path_save=results/weights/
```
- path: path to the dataset to use
- classifier: classifier to run
- split_per: split percentage for train and val sets
- seed: seed for splitting train, val sets (use small number)
- file: path to file that contains a specific split
- eval-true: whether to evaluate the model on test data after training
- path_save: path to save the trained classifier

To evaluate a classifier, run the following command:
```bash
python3 eval_feature_based.py --data=data/TSB_512/TSFRESH_TSB_512.csv --model=knn --model_path=results/weights/knn_512/ --path_save=results/raw_predictions/
```
- data: path to the time series to predict
- model: model to run
- model_path: path to the trained model
- path_save: path to save the results

## Model Selection Pipeline

We propose a benchmark and an evaluation of 16 time series classifiers used as model selection methods (with 12 anomaly detectors to be selected) applied on 16 datasets from different domains. Our pipeline can be summarized in the following figure.

<p align="center">
<img width="1000" src="./assets/figures/pipeline.jpg"/>
</p>

In the following section, we describe the datasets, anomaly detectors, and time series classification methods considered in our benchmark and evaluation.

### Datasets 

We first use the TSB-UAD benchmark (16 public datasets from heterogeneous domains).
Briefly, TSB-UAD includes the following datasets:

| Dataset    | Description|
|:--|:---------:|
|Dodgers| is a loop sensor data for the Glendale on-ramp for the 101 North freeway in Los Angeles and the anomalies represent unusual traffic after a Dodgers game.|
|ECG| is a standard electrocardiogram dataset and the anomalies represent ventricular premature contractions. We split one long series (MBA_ECG14046) with length ∼ 1e7) to 47 series by first identifying the periodicity of the signal.|
|IOPS| is a dataset with performance indicators that reflect the scale, quality of web services, and health status of a machine.|
|KDD21| is a composite dataset released in a recent SIGKDD 2021 competition with 250 time series.|
|MGAB| is composed of Mackey-Glass time series with non-trivial anomalies. Mackey-Glass time series exhibit chaotic behavior that is difficult for the human eye to distinguish.|
|NAB| is composed of labeled real-world and artificial time series including AWS server metrics, online advertisement clicking rates, real time traffic data, and a collection of Twitter mentions of large publicly-traded companies.|
|SensorScope| is a collection of environmental data, such as temperature, humidity, and solar radiation, collected from a typical tiered sensor measurement system.|
|YAHOO| is a dataset published by Yahoo labs consisting of real and synthetic time series based on the real production traffic to some of the Yahoo production systems.|
|Daphnet| contains the annotated readings of 3 acceleration sensors at the hip and leg of Parkinson’s disease patients that experience freezing of gait (FoG) during walking tasks.|
|GHL| is a Gasoil Heating Loop Dataset and contains the status of 3 reservoirs such as the temperature and level. Anomalies indicate changes in max temperature or pump frequency.|
|Genesis| is a portable pick-and-place demonstrator which uses an air tank to supply all the gripping and storage units.|
|MITDB| contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979.|
|OPPORTUNITY (OPP)| is a dataset devised to benchmark human activity recognition algorithms (e.g., classiffication, automatic data segmentation, sensor fusion, and feature extraction). The dataset comprises the readings of motion sensors recorded while users executed typical daily activities.|
|Occupancy| contains experimental data used for binary classiffication (room occupancy) from temperature, humidity, light, and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.|
|SMD (Server Machine Dataset)| is a 5-week-long dataset collected from a large Internet company. This dataset contains 3 groups of entities from 28 different machines.|
|SVDB| includes 78 half-hour ECG recordings chosen to supplement the examples of  supraventricular arrhythmias in the MIT-BIH Arrhythmia Database.|

The figure below shows some typical outliers in these datasets.

<p align="center">
<img width="1000" src="./assets/figures/display_data.jpg"/>
</p>

You may find more details (and the references) in the TSB-UAD benchmark [paper](https://www.paparrizos.org/papers/PaparrizosVLDB22a.pdf).

### Anomaly Detectors

We use 12 anomaly detection methods proposed for univariate time series. the following table lists and describes the methods considered:

| Anomaly Detection Method    | Description|
|:--|:---------:|
|Isolation Forest (IForest) | This method constructs the binary tree based on the space splitting and the nodes with shorter path lengths to the root are more likely to be anomalies. |
|The Local Outlier Factor (LOF)| This method computes the ratio of the neighboring density to the local density. |
|The Histogram-based Outlier Score (HBOS)| This method constructs a histogram for the data and the inverse of the height of the bin is used as the outlier score of the data point. |
|Matrix Profile (MP)| This method calculates as anomaly the subsequence with the most significant 1-NN distance. |
|NORMA| This method identifies the normal pattern based on clustering and calculates each point's effective distance to the normal pattern. |
|Principal Component Analysis (PCA)| This method projects data to a lower-dimensional hyperplane, and data points with a significant distance from this plane can be identified as outliers. |
|Autoencoder (AE)|This method projects data to the lower-dimensional latent space and reconstructs the data, and outliers are expected to have more evident reconstruction deviation. |
|LSTM-AD| This method build a non-linear relationship between current and previous time series (using Long-Short-Term-Memory cells), and the outliers are detected by the deviation between the predicted and actual values. |
|Polynomial Approximation (POLY)| This method build a non-linear relationship between current and previous time series (using polynomial decomposition), and the outliers are detected by the deviation between the predicted and actual values. |
| CNN | This method build a non-linear relationship between current and previous time series (using convolutional Neural Network), and the outliers are detected by the deviation between the predicted and actual values. |
|One-class Support Vector Machines (OCSVM)| This method fits the dataset to find the normal data's boundary. |

You may find more details (and the references) in the TSB-UAD benchmark [paper](https://www.paparrizos.org/papers/PaparrizosVLDB22b.pdf).

### Time Series Classification Algorithms

We consider 16 time series classification (TSC) algtorithms used as model selection. the following table lists and describes the methods considered:

| TSC  (as model seleciton)  | Description|
|:--|:---------:|
| SVC | maps training examples to points in space so as to maximize the gap between the two categories. |
| Bayes | uses Bayes’ theorem to predict the class of a new data point using the posterior probabilities for each class. |
| MLP | consists of multiple layers of interconnected neurons. |
| QDA | is a discriminant analysis algorithm for classification problems. |
| Adaboost | is a meta-algorithm using boosting technique with weak classifiers. |
| Descision Tree | is a tree-based approach that split data point into different leaves based on feature. |
| Random Forest  | is an ensemble Decision Trees fed with random sample (with replacement) of the training set and random set of features. |
| kNN | assigns the most common class among its k nearest neighbors. |
| Rocket | transforms input time series using a small set of convolutional kernels, and uses the transformed features to train a linear classifier. |
| ConvNet  | uses convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. |
| ResNet | is a ConvNet with residual connections between convolutional block. |
| InceptionTime | is a combination of ResNets with kernels of multiple sizes. |
| SIT-conv | is a transformer architecture with a convolutional layer as input. |
| SIT-linear | is a transformer architecture for which the time series are divided into non-overlapping patches and linearly projected into the embedding space. |
| SIT-stem | is a transformer architecture with convolutional layers with increasing dimensionality as input. |
| SIT-stem-ReLU | is similar to SIT-stem but with Scaled ReLU. |


## Overview of the results


We report in the following figure the average VUS-PR and inference time (i.e., predicting the detector to run and running it) for all detectors, the Oracle (the theoretical best model selection methods, in white), the Averaging Ensembling (in green), and the best time series classification used as model selection (in red). 

<p align="center">
<img width="500" src="./assets/figures/intro_fig.jpg"/>
</p>

This figure and many others comparisons are described (and reproductible) in [these notebook](https://github.com/boniolp/MSAD/tree/main/experiments/accuracy_analysis), summarized in [this document](TODO).
The overall accuracy tables (for VUS-PR and AUC-PR) are [here](https://github.com/boniolp/MSAD/tree/main/results/accuracy), and the execution time tables (training, prediction, and inference) are [here](https://github.com/boniolp/MSAD/tree/main/results/execution_time).
