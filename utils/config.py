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
TSB_acc_tables_path = "data/TSB/acc_tables/"

save_done_training = 'results/done_training/'	# when a model is done training a csv with training info is saved here
path_save_results = 'results/raw_predictions'	# when evaluating a model, the predictions will be saved here

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
