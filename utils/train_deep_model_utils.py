########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : train_deep_model_utils
#
########################################################################


import os
from pathlib import Path
import copy
import argparse
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import perf_counter, process_time
from datetime import datetime



class ModelExecutioner:
	def __init__(
		self,
		model,
		model_name,
		device='cuda',
		criterion=nn.CrossEntropyLoss(),
		use_scheduler=False,
		n_warmup_steps=4000,
		d_model=256,
		learning_rate=0.00001,
		runs_dir='runs',
		weights_dir='weights'
	):                      
		self.model = model
		self.runs_dir = runs_dir
		self.weights_dir = weights_dir
		self.model_name = model_name
		self.device = device
		self.criterion = criterion.to(self.device)
		self.use_scheduler = use_scheduler
		self.optimizer = torch.optim.Adam(
			self.model.parameters(), 
			lr=learning_rate, 
			betas=(0.9, 0.98), 
			eps=1e-9
		)
		self.n_warmup_steps = n_warmup_steps
		self.d_model = d_model
		self.training_time_epoch = 0
		self.epoch_best = 0
		self.learning_rate = learning_rate


	def train_one_epoch(self, epoch_index, tb_writer, training_loader):
		all_loss = []
		all_acc = []

		loop = tqdm(
			enumerate(training_loader), 
			total=len(training_loader),
			desc="Epoch [{}/{}]".format(epoch_index, self.n_epochs),
			leave=False,
			unit="batch",
			disable=not self.verbose
		)
		for i, (inputs, labels) in loop:
			# Move data to the same device as model
			inputs = inputs.to(self.device, dtype=torch.float32)
			labels = labels.to(self.device, dtype=torch.float32)

			# Zero the gradients for every batch
			if self.use_scheduler:
				self.scheduler.zero_grad()
			else:
				self.optimizer.zero_grad()

			# Make predictions for this batch
			outputs = self.model(inputs.float()).to(self.device)
			
			# Compute the loss and the gradients
			loss = self.criterion(outputs.float(), labels.long())
			loss.backward()

			# Adjust learning weights
			if self.use_scheduler:
				self.scheduler.step_and_update_lr()
			else:
				self.optimizer.step()

			# Compute the accuracy
			_, predictions = torch.max(outputs, 1)
			correct = (predictions.to(self.device) == labels.to(self.device)).sum().item()
			accuracy = correct / labels.size(0)

			
			# Gather data and report
			all_loss.append(loss.item())
			all_acc.append(accuracy)
			if i % 100 == 99:
				tb_x = epoch_index * len(training_loader) + i + 1
				tb_writer.add_scalar('Accuracy/train', np.mean(all_acc), tb_x)
				tb_writer.add_scalar('Loss/train', np.mean(all_loss), tb_x)
				tb_writer.flush()
					
			# Report on progress bar
			if i % 10 == 9:
				loop.set_postfix(
					acc=np.mean(all_acc),
					loss=np.mean(all_loss),
				)

		return np.mean(all_loss), np.mean(all_acc)


	def evaluate(self, dataloader):
		all_loss = []
		all_acc = []
		all_acc_top_k = []
		 
		# Switch model to eval mode
		self.model.eval()

		# The loop through batches
		with torch.no_grad():
			loop = tqdm(
				enumerate(dataloader), 
				total=len(dataloader), 
				desc="  validation: ",
				unit="batch",
				leave=False,
				disable=not self.verbose,
			)
			for i, (inputs, labels) in loop:
				# Move data to the same device as model
				inputs = inputs.to(self.device, dtype=torch.float32)
				labels = labels.to(self.device, dtype=torch.float32)

				# Make predictions for this batch
				outputs = self.model(inputs.float()).to(self.device)

				# Compute the loss
				loss = self.criterion(outputs.float(), labels.long())
				
				# Compute top k accuracy
				acc_top_k = self.compute_topk_acc(outputs, labels, k=4)

				all_loss.append(loss.item())
				all_acc.append(acc_top_k[1])
				all_acc_top_k.append(acc_top_k)

				# Report on progress bar
				if i % 10 == 9:
					loop.set_postfix(
						val_loss=np.mean(all_loss),
						val_acc=np.mean(all_acc),
					)

		return np.mean(all_loss), np.mean(all_acc), all_acc_top_k


	def train(self, n_epochs, training_loader, validation_loader, verbose=True):
		timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
		writer = SummaryWriter(self.runs_dir + '/{}_{}'.format(self.model_name, timestamp))
		self.n_epochs = n_epochs
		self.verbose = verbose
		best_val_loss = np.Inf
		best_val_acc = 0
		best_model = None

		# Set up early stop
		early_stopper = EarlyStopper(patience=50, min_delta=0)

		# Set up scheduler
		if self.use_scheduler:
			self.scheduler = ScheduledOptim(
				self.optimizer,
				lr_mul=.75,
				d_model=self.d_model,
				n_warmup_steps=self.n_warmup_steps,
			)

		# Check if saving dirs exist (if not create them)
		model_path = os.path.join(
			self.weights_dir, 
			self.model_name,
			'model_{}'.format(timestamp)
		)
		Path(os.path.join(self.weights_dir, self.model_name))\
		   .mkdir(parents=True, exist_ok=True)

		tic = perf_counter()
		for epoch in range(n_epochs):
			# Make sure gradient tracking is on and do a pass
			self.model.train(True)
			avg_loss, avg_acc = self.train_one_epoch(epoch, writer, training_loader)

			# We don't needs gradients on to do reporting
			self.model.train(False)

			# Run model on validation data to evaluate
			avg_val_loss, avg_val_acc, val_topk_acc = self.evaluate(validation_loader)

			avg_val_top1 = np.mean([x[1] for x in val_topk_acc])
			avg_val_top2 = np.mean([x[2] for x in val_topk_acc])
			avg_val_top3 = np.mean([x[3] for x in val_topk_acc])
			avg_val_top4 = np.mean([x[4] for x in val_topk_acc])

			# Epoch reporting
			print(
				"Epoch [{}/{}] {:.2f}secs : acc: {:.3f}, val_acc: {:.3f}, loss: {:.3f}, val_loss: {:.3f}, top k val_acc: k=1: {:.3f} k=2: {:.3f} k=3: {:.3f} k=4: {:.3f}"\
				.format(epoch, n_epochs, perf_counter()-tic, avg_acc, avg_val_acc, avg_loss, avg_val_loss, avg_val_top1, avg_val_top2, avg_val_top3, avg_val_top4)
			)
			
			# Log the running loss averaged per batch
			writer.add_scalars('Training vs. Validation Accuracy',
				{'Training': avg_acc, 'Validation': avg_val_acc},
				epoch + 1
			)
			writer.add_scalars('Training vs. Validation Loss',
				{'Training': avg_loss, 'Validation': avg_val_loss},
				epoch + 1
			)
			writer.flush()

			# Track best performance and save the model's state
			if avg_val_acc > best_val_acc:
				best_val_acc = avg_val_acc
				best_model = copy.deepcopy(self.model)
				torch.save(self.model.state_dict(), model_path)                                

			# Early stopping
			if (epoch > 3 and early_stopper.early_stop_acc(avg_val_acc)) or ((perf_counter()-tic) > 70000):
				break
		# Collect the results
		results = {
			'n_epochs': epoch + 1,
			'training_time': perf_counter()-tic,
			'acc': avg_acc, 
			'val_acc': avg_val_acc,
			'loss': avg_loss,
			'val_loss': avg_val_loss,
			'top_2_val_acc': avg_val_top2,
			'top_3_val_acc': avg_val_top3,
			'top_4_val_acc': avg_val_top4,
		}

		return best_model, results


	def compute_topk_acc(self, outputs, labels, k=4):
			'''Compute top k accuracy'''
			mean_acc_top_k = {k:[] for k in range(1, k+1)}

			_, y_pred = outputs.topk(k=k, dim=1)  # _, [B, n_classes] -> [B, maxk]
			y_pred = y_pred.t()
			target_reshaped = labels.view(1, -1).expand_as(y_pred)
			correct = (y_pred == target_reshaped)

			for k in range(1, k+1):
				ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
				flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
				tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
				topk_acc = tot_correct_topk / labels.size(0)  # topk accuracy for entire batch
				mean_acc_top_k[k].append(topk_acc.item())

			return mean_acc_top_k


	def torch_devices_info(self):
			print("----------------------------------------------------------------")
			print("Is there a GPU available: {}".format(torch.cuda.is_available()))
			print("Number of allocated devices: {}".format(torch.cuda.device_count()))
			curr_device_id = torch.cuda.current_device()
			print("Index of current device: {}".format(curr_device_id))
			print("Name of current divice: '{}'".format(torch.cuda.get_device_name(curr_device_id)))
			print("Memory allocated:", round(torch.cuda.memory_allocated(curr_device_id)/1024**3, 3), 'GB')
			print("Memory cached:   ", round(torch.cuda.memory_reserved(curr_device_id)/1024**3, 3), 'GB')
			print("----------------------------------------------------------------")

class EarlyStopper:
	def __init__(self, patience=5, min_delta=0.0001):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_val_loss = np.inf
		self.max_val_acc = 0

	def early_stop(self, val_loss):
		if val_loss < self.min_val_loss:
			self.min_val_loss = val_loss
			self.counter = 0
		elif val_loss > (self.min_val_loss - self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False

	def early_stop_acc(self, val_acc):
		if val_acc > self.max_val_acc:
			self.max_val_acc = val_acc
			self.counter = 0
		elif val_acc < (self.max_val_acc + self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False


class ScheduledOptim():
	'''A simple wrapper class for learning rate scheduling
					https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
	'''

	def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
		self._optimizer = optimizer
		self.lr_mul = lr_mul
		self.d_model = d_model
		self.n_warmup_steps = n_warmup_steps
		self.n_steps = 0


	def step_and_update_lr(self):
		"Step with the inner optimizer"
		self._update_learning_rate()
		self._optimizer.step()


	def zero_grad(self):
		"Zero out the gradients with the inner optimizer"
		self._optimizer.zero_grad()


	def _get_lr_scale(self):
		d_model = self.d_model
		n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
		return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


	def _update_learning_rate(self):
		''' Learning rate scheduling per step '''

		self.n_steps += 1
		lr = self.lr_mul * self._get_lr_scale()

		for param_group in self._optimizer.param_groups:
			param_group['lr'] = lr
		
		return lr

	def plot_lr(self, steps=400000):
		lr = []
		tmp_n_steps = self.n_steps
		self.n_steps = 0

		for i in range(steps):
			lr.append(self._update_learning_rate())
		
		plt.figure(figsize=(10, 8))
		plt.grid(True)
		plt.title('Scheduler d_model = {}'.format(self.d_model))
		plt.plot(lr)
		plt.ylabel('Learning Rate')
		plt.xlabel('Train Step')
		plt.tight_layout()
		plt.show()

		self.n_steps = tmp_n_steps


def json_file(x):
	if not os.path.isfile(x):
		raise argparse.ArgumentTypeError("{} is not a file".format(x))

	try:
		with open(x) as f:
   			variables = json.load(f)
	except Exception as e:
		raise argparse.ArgumentTypeError("{} is not a json file".format(x))

	return variables