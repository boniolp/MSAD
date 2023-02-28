########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : DiNO -> LIPADE -> Universite Paris Cite
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : config
#
########################################################################

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
