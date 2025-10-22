import os, torch

DIR_PATH = os.path.dirname(__file__)
dataset_name = "cifar10"
model_name = "mobilenet"
dataset_config = {"cifar10": {"input_dim": 330, "dim": 300, "n_classes": 10}}
dataset_path = os.path.join("datasets")
seed = 42 # the answer to life the universe and everything
cuda = True
distribution = "linear" 
exit_type = "bnpool"
batch_size_train = 256
batch_size_test = 1
pretrained = True
split_ratio = 0.2
