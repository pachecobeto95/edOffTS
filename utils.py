import config
from torchvision import datasets, transforms
import torch, os, sys, requests, cv2
import numpy as np

def load_cifar10():
    """
    Função para carregar o dataset CIFAR-10 com transformações básicas.
    
    Args:
        batch_size (int): Tamanho do lote para os DataLoaders.
        data_dir (str): Diretório onde os dados serão armazenados ou lidos.
    
    Returns:
        tuple: (train_loader, test_loader) DataLoaders para treino e teste.
    """
    # Transformações para normalizar os dados e realizar augmentations simples
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Flip horizontal aleatório
        transforms.RandomCrop(32, padding=4),  # Crop com padding
        transforms.ToTensor(),  # Converte imagem para tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalização
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # Carregar os datasets
    train_dataset = datasets.CIFAR10(
        root=config.dataset_path, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=config.dataset_path, train=False, download=True, transform=transform_test
    )
    
    # Criar DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size_train, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, test_dataset.classes
