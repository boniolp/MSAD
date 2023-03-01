########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : data_loader
#
########################################################################

import os, glob

import numpy as np
import pandas as pd
import time
from tqdm import tqdm


class DataLoader:
	"""This class is used to read and load data from the benchmark.
	When the object is created the path to the benchmark directory
	should be given.
	"""

	def __init__(self, data_path):
		self.data_path = data_path


	def get_dataset_names(self):
		'''Returns the names of existing datasets. 
		Careful, this function will not return any files in the given
		directory but only the names of the sub-directories
		as they are the datasets (not the timeseries).

		:return: list of datasets' names (list of strings)
		'''
		names = os.listdir(self.data_path)

		return [x for x in names if os.path.isdir(os.path.join(self.data_path, x))]
		

	def load(self, dataset):
		'''
		Loads the specified datasets

		:param dataset: list of datasets
		:return x: timeseries
		:return y: corresponding labels
		:return fnames: list of names of the timeseries loaded
		'''
		x = []
		y = []
		fnames = []


		if not isinstance(dataset, list):
			raise ValueError('only accepts list of str')

		pbar = tqdm(dataset)
		for name in pbar:
			pbar.set_description('Loading ' + name)
			for fname in glob.glob(os.path.join(self.data_path, name, '*.out')):
				curr_data = pd.read_csv(fname, header=None).to_numpy()
				
				if curr_data.ndim != 2:
					raise ValueError('did not expect this shape of data: \'{}\', {}'.format(fname, curr_data.shape))

				# Skip files with no anomalies
				if not np.all(curr_data[0, 1] == curr_data[:, 1]):
					x.append(curr_data[:, 0])
					y.append(curr_data[:, 1])
					# Remove path from file name, keep dataset, time series name
					fname = '/'.join(fname.split('/')[-2:])		
					fnames.append(fname.replace(self.data_path, ''))
					
		return x, y, fnames


	def load_df(self, dataset):
		'''
		Loads the time series of the given datasets and returns a dataframe

		:param dataset: list of datasets
		:return df: a single dataframe of all loaded time series
		'''
		df_list = []
		pbar = tqdm(dataset)

		if not isinstance(dataset, list):
			raise ValueError('only accepts list of str')

		for name in pbar:
			pbar.set_description(f'Loading {name}')
			
			for fname in glob.glob(os.path.join(self.data_path, name, '*.csv')):
				curr_df = pd.read_csv(fname, index_col=0)
				curr_index = [os.path.join(name, x) for x in list(curr_df.index)]
				curr_df.index = curr_index

				df_list.append(curr_df)
				
		df = pd.concat(df_list)

		return df


	def load_timeseries(self, timeseries):
		'''
		Loads specified timeseries

		:param fnames: list of file names
		:return x: timeseries
		:return y: corresponding labels
		:return fnames: list of names of the timeseries loaded
		'''
		x = []
		y = []
		fnames = []

		for fname in tqdm(timeseries, desc='Loading timeseries'):
			curr_data = pd.read_csv(os.path.join(self.data_path, fname), header=None).to_numpy()
			
			if curr_data.ndim != 2:
				raise ValueError('did not expect this shape of data: \'{}\', {}'.format(fname, curr_data.shape))

			# Skip files with no anomalies
			if not np.all(curr_data[0, 1] == curr_data[:, 1]):
				x.append(curr_data[:, 0])
				y.append(curr_data[:, 1])
				fnames.append(fname)

		return x, y, fnames
