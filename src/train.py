import torch
import torch.nn as nn
import torch.optim as optim
from data.download_dataset import download_cifar10
from src.utils.data_loader import get_dataloader
from src.utils.visualization import show_generated_images
from src.models.dcgan import Generator,Discriminator

# Config..

z_dim = 100
batch_size = 64
lr = 0.0002
num_epoch = 5
img_channels = 3
features_g = 64
features_d = 64


# Load dataset
dataset = download_cifar10()
data_loader = get_dataloader(dataset,batch_size)