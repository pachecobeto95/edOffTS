import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import argparse, config, torch, os, ee_dnns, utils, sys, ee_nn_calibration, spsa
from tqdm import tqdm
from scipy.stats import beta




def exp_acc_edge(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes):

	n_samples = len(df_edge)

	# --- Compute calibrated confidences per branch ---
	calib_confs = calibrating_confs(temp_list, n_branches, df_edge, n_classes)

	# --- Initialize masks ---
	remaining_mask = np.ones(n_samples, dtype=bool)
	correct_edge, total_edge = 0, 0

	# --- Process all early exits (excluding final branch) ---
	for i in range(n_branches):
		# Select samples that exit at this branch
		early_exit_mask = (calib_confs[i] >= threshold) & remaining_mask

		n_exit = early_exit_mask.sum()
		if n_exit == 0:
			continue

		correctness = df_edge[f"correct_branch_{i+1}"].to_numpy()
		correct_edge += correctness[early_exit_mask].sum()
		total_edge += n_exit

		# Remove exited samples from future branches
		remaining_mask &= ~early_exit_mask

	exp_acc = correct_edge / total_edge if (total_edge > 0) else 0.0
	ee_prob = total_edge / n_samples

	return exp_acc, ee_prob



def theo_beta_function(temp_list, n_branches, threshold, df_edge, df_cloud, beta, overhead, n_classes):

	inf_time, _, _ = compute_inference_time(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes)

	#The following line computes the on-device accuracy using our theoretical model
	acc, ee_prob = theoretical_accuracy_edge(temp_list, n_branches, threshold, df_edge, overhead, n_classes)
	
	f = inf_time - beta*acc

	exp_acc, _ = exp_acc_edge(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes)

	print("Acc Device: %s"%acc)
	print("Acc Exp: %s"%(exp_acc))

	return f, ee_prob


def calibrating_confs(temp_list, n_branches, df_edge, n_classes):

	# --- Compute calibrated confidences per branch ---
	calib_confs = {}
	for i in range(n_branches):
		logits = np.stack([df_edge[f"logit_class_{c+1}_branch_{i+1}"] for c in range(n_classes)], axis=1)
		scaled_logits = logits / temp_list[i]
		exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
		probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
		calib_confs[i] = np.max(probs, axis=1)

	return calib_confs


def theoretical_accuracy_edge(temp_list, n_branches, threshold, df_edge, overhead, n_classes):

	calib_confs = calibrating_confs(temp_list, n_branches, df_edge, n_classes)

	n_samples = len(df_edge)

	#The following line computes the term P[f_{L-1}^{T_{L-1}} >= gamma]
	ee_prob_edge = np.sum(calib_confs[n_branches-1] >= threshold) / n_samples

	sum_success_prob = 0

	for i in range(n_branches):
		success_prob = compute_success_probability(df_edge, temp_list, i, calib_confs[i], threshold)
		sum_success_prob += success_prob

	#print(sum_success_prob, ee_prob_edge)
	acc_edge = sum_success_prob/ee_prob_edge if (ee_prob_edge > 0) else 0.0
	#print(acc_edge)

	return acc_edge, ee_prob_edge


def estimate_joint_prob_conditional(calib_confs, threshold, f_values, n_bins=100):
	"""
	Estimate p(f_prev < gamma, f_curr = f) without assuming independence using conditional Beta distributions.

	Parameters
	----------
	f_prev : np.ndarray
		Confidences from previous branch (f_{l-1})
	f_curr : np.ndarray
		Confidences from current branch (f_l)
	threshold : float
		Threshold for f_prev
	f_values : np.ndarray
		Points at which to evaluate the joint probability
	n_bins : int
		Number of bins to discretize f_prev

	Returns
	-------
	joint_prob : np.ndarray
	Estimated joint probability at each f in f_values
	"""
	a, b, loc, scale = beta.fit(calib_confs, floc=0, fscale=1)
	joint_prob = beta.pdf(f_values, a, b, loc=loc, scale=scale)
	return joint_prob

def compute_success_probability(df_edge, temp_list, i_branch, calib_confs, threshold, step=0.001):

	correctness = df_edge[f"correct_branch_{i_branch+1}"].to_numpy(dtype=int)

	f_values = np.arange(threshold, 1.0 + step, step)
	expectations = np.zeros_like(f_values)

	integral_sum = 0.0

	for j, f in enumerate(f_values):
		mask = (calib_confs >= f - step/2) & (calib_confs < f + step/2)

		joint_prob = estimate_joint_prob_conditional(calib_confs, threshold, f)

		if np.any(mask):
			exp_val = np.mean(correctness[mask])
		else:
			exp_val = 0.0
		expectations[j] = exp_val*joint_prob
		integral_sum += exp_val *joint_prob * step

	#return integral_sum, f_values, expectations
	return integral_sum


def compute_inference_time(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes):
	"""
	Compute the average inference time and early-exit probability across multiple branches 
	of an early-exit DNN, considering temperature scaling calibration and cloud overhead.
	"""

	n_exits = n_branches + 1

	n_samples = len(df_edge)
	avg_inference_time = 0.0
	n_exit_per_branch = np.zeros(n_branches)
	total_exited = 0

	calib_confs = calibrating_confs(temp_list, n_branches, df_edge, n_classes)

	# --- Early-exit simulation on edge ---
	remaining_mask = np.ones(n_samples, dtype=bool)

	for i in range(n_branches):
		early_exit_mask = (calib_confs[i] >= threshold) & remaining_mask

		n_exit = early_exit_mask.sum()
		n_exit_per_branch[i] = n_exit
		total_exited += n_exit

		inf_time_branch_device = df_edge[f"cum_inf_time_branch_{i+1}"].mean()
		avg_inference_time += n_exit * inf_time_branch_device

		remaining_mask &= ~early_exit_mask

	# --- Offload to cloud ---
	n_exit_cloud = remaining_mask.sum()
	n_exit_per_branch[-1] = n_exit_cloud

	if n_exit_cloud > 0:
		edge_cum_time = df_edge[f"cum_inf_time_branch_{n_branches}"].mean()
		cloud_inf_time = df_cloud[f"cum_inf_time_branch_{n_branches+1}"].mean() - edge_cum_time

		# Add cloud overhead and processing time
		avg_inference_time += n_exit_cloud * (edge_cum_time + overhead + cloud_inf_time)

	# --- Normalize ---
	avg_inference_time /= float(n_samples)
	early_exit_prob = total_exited / float(n_samples)

	'''
	# Usa a normalização Min-Max
	time_range = T_max - T_min

	if time_range <= 0:
		# Caso degenerado (tempos min e max iguais)
		avg_inference_time_norm = 1.0 
	else:
		avg_inference_time_norm = (avg_inference_time - T_min) / time_range
	'''

	return avg_inference_time, early_exit_prob, n_exit_per_branch



class EE_Calibration(object):
	def __init__(self, args, df_inf_edge, df_inf_cloud, threshold, overhead, beta):
		self.args = args
		self.df_inf_edge = df_inf_edge
		self.df_inf_cloud = df_inf_cloud
		self.threshold = threshold
		self.overhead = overhead
		self.beta = beta


	def calibrationEEDNN_SPSA(self):
		temp_list = np.ones(10)
		optim_spsa = spsa.SPSA(self.args, theo_beta_function, temp_list, self.df_inf_edge, self.df_inf_cloud, self.threshold, self.overhead, self.beta)
		return optim_spsa.min()
