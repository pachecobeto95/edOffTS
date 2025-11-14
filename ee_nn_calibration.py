import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import argparse, config, torch, os, ee_dnns, utils, sys, ee_nn_calibration, spsa, ts
from tqdm import tqdm
from scipy.stats import beta
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression


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
	ee_prob = total_edge / n_samples if (total_edge > 0) else 0.0

	return exp_acc, ee_prob



def theo_beta_function(temp_list, n_branches, threshold, df_edge, df_cloud, beta, overhead, n_classes):

	inf_time, ee_prob, _ = compute_inference_time(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes)
	inf_time = 1000*inf_time

	#The following line computes the on-device accuracy using our theoretical model
	acc, _ = theoretical_accuracy_edge(temp_list, n_branches, threshold, df_edge, overhead, n_classes)
	
	exp_acc, exp_ee_prob = exp_acc_edge(temp_list, n_branches, threshold, df_edge, df_cloud, overhead, n_classes)


	f = inf_time - beta*acc

	#print("Acc Device: %s"%acc)
	#print("Acc Exp: %s"%(exp_acc), exp_ee_prob)
	#print("Inf Time: %s"%(inf_time), a)
	#print(temp_list)

	return f, inf_time, acc, ee_prob


# ----------------------------------------------------------------------
# 1. FUNÇÕES DE SUPORTE (PDF e Suavização de Acurácia Local)
# ----------------------------------------------------------------------

def estimate_joint_prob_conditional_KDE(calib_confs, threshold, f_values, n_bins=100):
	"""
	Estima a PDF usando Kernel Density Estimation (KDE) nos dados de confiança.
	"""
	eps = 1e-6
	data = np.clip(calib_confs, eps, 1 - eps)

	try:
		# Cria o estimador KDE (usando bandwidth automática 'scott' ou 'silverman')
		kde = gaussian_kde(data)
	except ValueError:
		# Caso de dados insuficientes
		return np.ones_like(f_values) / len(f_values)

	# Avalia a PDF nos pontos desejados
	joint_prob = kde.evaluate(f_values)

	# Normaliza a integral para 1.0
	integral_kde = np.trapz(joint_prob, f_values)
	if integral_kde > 0:
		joint_prob /= integral_kde
	
	return joint_prob

def compute_local_accuracy_smoother(calib_confs_i, correctness_i, f_values):
	"""
	Estima E[Acerto | f_i = f] usando Regressão Logística (suavização da curva de calibração).
	"""
	# Mínimo de amostras necessário para treinar a regressão (ex: 20)
	if len(calib_confs_i) < 20: 
		return None 

	try:
		# X é a confiança, Y é a correção (target binário: 0 ou 1)
		X = calib_confs_i.reshape(-1, 1)
		Y = correctness_i
		
		# Regressão Logística para modelar P(Acerto | Confiança)
		# C alto para um ajuste mais próximo (Platt Scaling implícito)
		model = LogisticRegression(solver='liblinear', C=1e6) 
		model.fit(X, Y)
		
		# Prediz a probabilidade de acerto (exp_val) para cada ponto f_values
		f_values_reshaped = f_values.reshape(-1, 1)
		# P(y=1|f) é o segundo índice (índice 1) de predict_proba
		exp_val_smoothed = model.predict_proba(f_values_reshaped)[:, 1] 
		
		return exp_val_smoothed
		
	except Exception:
		# Em caso de erro (ex: classes desbalanceadas ou constantes), retorna None
		return None

# ----------------------------------------------------------------------
# 2. FUNÇÕES DE CÁLCULO DE PROBABILIDADE E ACURÁCIA (HÍBRIDA)
# ----------------------------------------------------------------------

# Mantendo a função calibrating_confs, pois é necessária
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


def compute_success_probability_hybrid(df_edge, temp_list, i_branch, calib_confs_all, threshold, n_branches, n_classes, step=0.001):
	
	# 1. Identificar amostras que chegam ao branch i (Condição Empírica)
	remaining_mask = np.ones(len(df_edge), dtype=bool)
	for j in range(i_branch):
		remaining_mask &= ~(calib_confs_all[j] >= threshold)

	data_subset = df_edge[remaining_mask].copy() # Usar .copy() para evitar SettingWithCopyWarning
	
	if len(data_subset) == 0:
		return 0.0, 0.0

	calib_confs_i = calib_confs_all[i_branch][remaining_mask]
	correctness_i = data_subset[f"correct_branch_{i_branch+1}"].to_numpy(dtype=int)
	
	# 2. Setup para a Integral
	f_values = np.arange(threshold, 1.0 + step, step)
	integral_sum_conditional = 0.0
	
	# Estima a PDF do subconjunto que chegou
	joint_prob_array = estimate_joint_prob_conditional_KDE(calib_confs_i, threshold, f_values)
	
	# Suaviza o valor esperado de acerto (E[Acerto | f_i])
	exp_val_smoothed_array = compute_local_accuracy_smoother(calib_confs_i, correctness_i, f_values)

	# 3. Cálculo da Integral
	for j, f in enumerate(f_values):
		
		if exp_val_smoothed_array is not None:
			# Usa o valor suavizado da regressão
			exp_val = exp_val_smoothed_array[j]
		else:
			# Fallback para o valor empírico ruidoso
			mask = (calib_confs_i >= f - step/2) & (calib_confs_i < f + step/2)
			exp_val = np.mean(correctness_i[mask]) if np.any(mask) else 0.0

		joint_prob = joint_prob_array[j]
		
		# Elemento da integral: E[Acerto|f_i] * P(f_i|Chegou em i) * df
		integral_sum_conditional += exp_val * joint_prob * step
		
	# 4. Combinação Híbrida Final
	prob_exit_given_arrived = np.sum(joint_prob_array * step)
	prob_arrived = len(data_subset) / len(df_edge)
	
	# Numerador (Success Prob): P_Teórico-Híbrido(Acerto E Exit no i)
	success_prob = prob_arrived * integral_sum_conditional
	
	# Denominador (Exit Prob): P_Teórico-Híbrido(Exit no i)
	prob_exit_i = prob_arrived * prob_exit_given_arrived

	return success_prob, prob_exit_i


def theoretical_accuracy_edge(temp_list, n_branches, threshold, df_edge, overhead, n_classes):

	# 1. Pré-calcula todas as confianças calibradas
	calib_confs_all = calibrating_confs(temp_list, n_branches, df_edge, n_classes)

	sum_success_prob = 0.0 # Numerador: Sum P(Exit e Acerto)
	sum_exit_prob = 0.0    # Denominador: Sum P(Exit)
	
	for i in range(n_branches):
		# Usa a nova função híbrida com suavização
		success_prob_i, prob_exit_i = compute_success_probability_hybrid(
			df_edge, temp_list, i, calib_confs_all, threshold, n_branches, n_classes
		)
		sum_success_prob += success_prob_i
		sum_exit_prob += prob_exit_i 

	# O termo ee_prob é a probabilidade total de saída no edge (Denominador)
	ee_prob = sum_exit_prob
	
	# Acc_Edge = Sum P(Exit e Acerto) / Sum P(Exit)
	acc_edge = sum_success_prob / ee_prob if (ee_prob > 0) else 0.0

	return acc_edge, ee_prob

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
		#cloud_inf_time = df_cloud[f"cum_inf_time_branch_{n_branches+1}"].mean() - edge_cum_time
		cloud_inf_time = df_cloud[f"delta_inf_time_branch_{n_branches+1}"].mean()

		# Add cloud overhead and processing time
		avg_inference_time += n_exit_cloud * (edge_cum_time + overhead + cloud_inf_time)

	#print(edge_cum_time, cloud_inf_time, (edge_cum_time + overhead + cloud_inf_time))
	#print(total_exited, n_exit_cloud)
	#sys.exit()

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


	def calibrationEEDNN_SPSA(self, temp_list):
		optim_spsa = spsa.SPSA(self.args, theo_beta_function, temp_list, self.df_inf_edge, 
			self.df_inf_cloud, self.threshold, self.overhead, self.beta)
		
		return optim_spsa.min()

	def calibrationEEDNN_TS(self, temp_list):

		optim_ts = ts.TemperatureScaler(self.args, temp_list, self.df_inf_edge, 
			self.df_inf_cloud, self.threshold, self.overhead, self.beta)
		
		return optim_ts.min()


	def calibrationEEDNN_NO_CALIB(self, temp_list):

		n_classes = config.dataset_config[self.args.dataset_name]["n_classes"]

		loss, inf_time, exp_acc, ee_prob = ts.exp_beta_function(temp_list, self.args.n_branches, 
			self.threshold, self.df_inf_edge, self.df_inf_cloud, self.beta, self.overhead, n_classes)

		print("No Optimization completed. Best Loss: %s, Acc Edge: %s, Inf Time: %s, EE Prob: %s"
			%(loss, exp_acc, inf_time, ee_prob))
		
		return temp_list, loss, inf_time, exp_acc, ee_prob