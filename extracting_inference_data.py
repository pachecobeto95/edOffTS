import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import argparse, config, torch, os, ee_dnns, utils, sys
from tqdm import tqdm


def extracting_ee_inference_data(args, class_names, test_loader, model, device):

	n_exits = args.n_branches + 1	
	conf_list, correct_list, delta_inf_time_list, cum_inf_time_list = [], [], [], []
	prediction_list, target_list, class_name_list, logits_list = [], [], [], []

	model.eval()

	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			class_name = class_names[target.item()]

			# Obtain confs and predictions for each side branch.
			logits, conf_branches, predictions_branches, delta_inf_time_branches, cum_inf_time_branches = model.forwardExtractingInferenceData(data)

			logits_list.append(logits), conf_list.append([conf_branch.item() for conf_branch in conf_branches])
			delta_inf_time_list.append(delta_inf_time_branches), cum_inf_time_list.append(cum_inf_time_branches)

			correct_list.append([predictions_branches[i].eq(target.view_as(predictions_branches[i])).sum().item() for i in range(n_exits)])
			target_list.append(target.item()), prediction_list.append([predictions_branches[i].item() for i in range(n_exits)])

			class_name_list.append(class_name)

	conf_list, correct_list, delta_inf_time_list = np.array(conf_list), np.array(correct_list), np.array(delta_inf_time_list)
	cum_inf_time_list, prediction_list = np.array(cum_inf_time_list), np.array(prediction_list)
	logits_list = np.array(logits_list)

	accuracy_branches = [sum(correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)]

	#print("Accuracy: %s"%(accuracy_branches))
	result_dict = {"device": len(target_list)*[str(device)],
	"target": target_list, "class_name": class_name_list}

	for i in range(n_exits):
		result_dict["conf_branch_%s"%(i+1)] = conf_list[:, i]
		result_dict["correct_branch_%s"%(i+1)] = correct_list[:, i]
		result_dict["delta_inf_time_branch_%s"%(i+1)] = delta_inf_time_list[:, i]
		result_dict["cum_inf_time_branch_%s"%(i+1)] = cum_inf_time_list[:, i]
		result_dict["prediction_branch_%s"%(i+1)] = prediction_list[:, i]
		for j_class in range(len(class_names)):
			result_dict["logit_class_%s_branch_%s"%(j_class+1, i+1)] = logits_list[:, i, j_class]

	print(len(result_dict["conf_branch_1"]), len(result_dict["logit_class_1_branch_1"]))

	#Converts to a DataFrame Format.
	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df


def main(args):

	n_classes = config.dataset_config[args.dataset_name]["n_classes"]

	device = torch.device('cuda' if ((torch.cuda.is_available()) and (args.location == "desktop")) else 'cpu')

	model_path = os.path.join(config.DIR_PATH, "models", "ee_model_%s_%s_branches_%s.pth"%(args.model_name, 
		args.n_branches, args.dataset_name))

	inf_data_dir_path = os.path.join(config.DIR_PATH, "inference_data")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_%s.csv"%(args.model_name, 
		args.n_branches, args.dataset_name, args.location))
	
	ee_model = ee_dnns.load_eednn_model(args, n_classes, model_path, device)

	_, test_loader, class_names = utils.load_cifar10()

	df_inf_data = extracting_ee_inference_data(args, class_names, test_loader, ee_model, device)

	df_inf_data.to_csv(inf_data_path, mode='a', header=not os.path.exists(inf_data_path))



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

	parser.add_argument('--location', type=str, help='Which machine extracts the inference data', choices=["laptop", "desktop"],
		default="desktop")

	args = parser.parse_args()

	main(args)