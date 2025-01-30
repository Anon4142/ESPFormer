import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from vit import ViT
from tqdm import tqdm
from torch.utils.data import Subset
import wandb

np.random.seed(0)

subset_perc = 50

# Data paths
train_dir = 'examples/data/train'
test_dir = 'examples/data/test1'

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))


labels = [path.split('/')[-1].split('.')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels)

print(f"Train Data: {len(train_list)}")

# Data transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Custom Dataset
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label

# 1, 10, 50      
train_data = CatsDogsDataset(train_list, transform=train_transforms)
subset_indices = np.random.choice(len(train_data), int((subset_perc / 100) * len(train_data)), replace=False)
print(f"Train Data subset: {len(subset_indices)}")
# print(subset_indices)

np.save(f'subset_indices_{subset_perc}.npy', subset_indices)