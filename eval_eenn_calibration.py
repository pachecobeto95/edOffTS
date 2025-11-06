import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import argparse, config, torch, os, ee_dnns, utils, sys, ee_nn_calibration
from tqdm import tqdm


def calibrating_eenn(args, df_inf_data_edge, df_inf_data_cloud, threshold, overhead, beta):

    calibration_method = f"calibrationEEDNN_{args.calibration_type.upper()}"

    c = ee_nn_calibration.EE_Calibration(args, df_inf_data_edge, df_inf_data_cloud, threshold, overhead, beta)

    calibration_func = getattr(c, calibration_method, None)

    if calibration_func is None:
        raise ValueError(f"Unsupported calibration method: {args.calibration_type}")

    return calibration_func()


def save_results(theta, loss, inf_time, acc, ee_prob, beta, overhead, threshold, filepath):
	"""
	Save the current experiment results into a DataFrame and append them to a CSV file.
	"""

	df_results = pd.DataFrame()

	# Ensure theta is a numpy array
	theta = np.array(theta)

	# Create a dictionary for the new row
	result_row = {"loss": loss, "inference_time": inf_time, "accuracy": acc,"ee_prob": ee_prob, "beta": beta, "overhead": overhead, "threshold": threshold}

	# Add theta components as individual columns
	for i, t in enumerate(theta):
		result_row[f"theta_{i+1}"] = t

	df_row = pd.DataFrame([result_row])

	# Check if the file already exists
	file_exists = os.path.isfile(filepath)

	# Append to file (write header only if the file doesn’t exist yet)
	df_row.to_csv(filepath, mode='a', index=False, header=not file_exists)



def main(args):

	n_classes = config.dataset_config[args.dataset_name]["n_classes"]

	device = torch.device('cuda' if ((torch.cuda.is_available()) and (args.location == "desktop")) else 'cpu')

	model_path = os.path.join(config.DIR_PATH, "models", "ee_model_%s_%s_branches_%s.pth"%(args.model_name, 
		args.n_branches, args.dataset_name))

	#ee_model = ee_dnns.load_eednn_model(args, n_classes, model_path, device)

	results_path = "results_%s.csv"%(args.calibration_type)

	inf_data_dir_path = os.path.join(config.DIR_PATH, "inference_data")

	inf_data_edge_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_laptop.csv"%(args.model_name, 
		args.n_branches, args.dataset_name))

	inf_data_cloud_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_desktop.csv"%(args.model_name, 
		args.n_branches, args.dataset_name))
	
	#_, test_loader, class_names = utils.load_cifar10()

	df_inf_data_edge, df_inf_data_cloud = pd.read_csv(inf_data_edge_path), pd.read_csv(inf_data_cloud_path)


	threshold_list = np.round(np.arange(0.5, 1.05, 0.05), 2)
	overhead_list = np.arange(0, 100, 5)
	beta_list = np.arange(0, 100, 5)

	temp_list = np.ones(n_classes)

	for threshold in threshold_list:

		for overhead in overhead_list:

			for beta in beta_list:
				theta, loss, inf_time, acc, ee_prob = calibrating_eenn(args, df_inf_data_edge, df_inf_data_cloud, threshold, overhead, beta)
				save_results(theta, loss, inf_time, acc, ee_prob, beta, overhead, threshold, results_path)

			#acc_edge = ee_nn_calibration.theoretical_accuracy_edge(temp_list, args.n_branches, threshold, df_inf_data_edge, df_inf_data_cloud, overhead, args.dataset_name)
			#acc_exp, _ = ee_nn_calibration.exp_acc_edge(temp_list, args.n_branches, threshold, df_inf_data_edge, df_inf_data_cloud, overhead, args.dataset_name)			
			#print(f"Threshold:{threshold}, Overhead: {overhead}, Inf time: {inf_time}, EE prob: {ee_prob}")
			#print(f"Threshold:{threshold}, Overhead: {overhead}, Acc Edge: {acc_edge}")
			#print(f"Threshold:{threshold}, Overhead: {overhead}, Acc EXP: {acc_exp}")


			#sys.exit()





if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech-256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	parser.add_argument('--n_branches', type=int, default=1, help='Number of side branches.')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--calibration_type', type=str, choices=['spsa'], help='Calibration Type')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--n_rounds', type=int, default=100, help='Calibration Type')


	args = parser.parse_args()

	main(args)