########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : config
#
########################################################################

from models.model.convnet import ConvNet
from models.model.inception_time import InceptionModel
from models.model.resnet import ResNetBaseline
from models.model.sit import SignalTransformer


# Important paths
TSB_data_path = "data/TSB/data/"
TSB_metrics_path = "data/TSB/metrics/"
TSB_scores_path = "data/TSB/scores/"


# Detector
detector_names = [
	'AE', 
	'CNN', 
	'HBOS', 
	'IFOREST', 
	'IFOREST1', 
	'LOF', 
	'LSTM', 
	'MP', 
	'NORMA', 
	'OCSVM', 
	'PCA', 
	'POLY'
]


# Dict of model names to Constructors
deep_models = {
	'convnet':ConvNet,
	'inception_time':InceptionModel,
	'inception':InceptionModel,
	'resnet':ResNetBaseline,
	'sit':SignalTransformer,
}
