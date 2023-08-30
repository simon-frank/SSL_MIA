#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######################## DATA PREPROCESSING AND LOADING ############################


##### Download link for the lung and colon histopathology dataset####
#https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af#
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader


# Function to create data loaders
def create_data_loaders(dataset, batch_size=16, train_ratio=0.8, num_workers=4, shuffle=True):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader

# Function to load the colon dataset
def load_colon_dataset(dataset_path):
    colon_dataset = ImageFolder(dataset_path, transform=ToTensor())
    return colon_dataset

# Function to load the lung dataset
def load_lung_dataset(dataset_path):
    lung_dataset = ImageFolder(dataset_path, transform=ToTensor())
    return lung_dataset


# Function to preprocess a dataset
def preprocess_dataset(dataset, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    preprocessed_data = []
    for image, _ in dataset:
        preprocessed_image = transform(image)
        preprocessed_data.append(preprocessed_image)

    return preprocessed_data

# Set the dataset paths
colon_dataset_path = "/Users/dr.elsherif/Downloads/lung_colon_image_set/colon_image_sets"
lung_dataset_path = "/Users/dr.elsherif/Downloads/lung_colon_image_set/lung_image_sets"

# Load and preprocess the datasets
colon_dataset = load_colon_dataset(colon_dataset_path)
preprocessed_colon_dataset = preprocess_dataset(colon_dataset)

lung_dataset = load_lung_dataset(lung_dataset_path)
preprocessed_lung_dataset = preprocess_dataset(lung_dataset)


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as pl
from torchvision.models import resnet50, resnet18
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader

# Define the backbone models

# ResNet-50 backbone model
class Base50(nn.Module):
    def __init__(self, readout_head):
        super().__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.readout_head = readout_head
        self.resize = Resize((224, 224))

    def forward(self, x):
        x = self.resize(x)
        x = x.permute(0, 3, 1, 2)  # Permute dimensions to [batch_size, channels, height, width]
        x = self.resnet50(x)
        x = self.readout_head(x)
        return x

# ResNet-18 backbone model
class Base18(nn.Module):
    def __init__(self, readout_head):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.readout_head = readout_head
        self.resize = Resize((224, 224))

    def forward(self, x):
        x = self.resize(x)
        x = x.permute(0, 3, 1, 2)  # Permute dimensions to [batch_size, channels, height, width]
        x = self.resnet18(x)
        x = self.readout_head(x)
        return x

# Define the readout head with adaptive pooling
class ReadoutHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the combined model
class Base(pl.LightningModule):
    def __init__(self, backbone, readout_head, input_shape, config):
        super().__init__()
        self.lr = config['finetuning']['lr']
        self.train_all = config['finetuning']['trainall']
        self.backbone = backbone
        self.readout_head = readout_head
        self.resize = nn.Upsample(size=(input_shape[1], input_shape[2]), mode='bilinear', align_corners=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.resize(x)
        x = x.permute(0, 3, 1, 2)  # Permute dimensions to [batch_size, channels, height, width]
        x = self.backbone(x)
        x = self.readout_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'] 
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Function to train the model
def train_model(model, train_loader, val_loader, max_epochs=10, accelerator=None):
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator)
    trainer.fit(model, train_loader, val_loader)


# Define the configuration dictionaries

# Configuration for the colon dataset
config_colon = {
    'finetuning': {'lr': 0.001, 'trainall': True},
    'num_classes': 2,  # Number of classes in the colon dataset
}

# Configuration for the lung dataset
config_lung = {
    'finetuning': {'lr': 0.001, 'trainall': True},
    'num_classes': 3,  # Number of classes in the lung dataset
}


# In[ ]:


# Create data loaders for colon training and validation
colon_train_loader, colon_val_loader = create_data_loaders(preprocessed_colon_dataset)

# Create data loaders for lung training and validation
lung_train_loader, lung_val_loader = create_data_loaders(preprocessed_lung_dataset)

# Create the readout head for colon dataset
readout_head_colon = ReadoutHead(in_features=1000, num_classes=2)

# Create the readout head for lung dataset
readout_head_lung = ReadoutHead(in_features=512, num_classes=3)

# Create the model for colon dataset using ResNet50 as the backbone
backbone_colon = Base50(readout_head_colon)
model_colon = Base(backbone_colon, readout_head_colon, colon_dataset[0][0].shape, config_colon)

# Create the model for lung dataset using ResNet18 as the backbone
backbone_lung = Base18(readout_head_lung)
model_lung = Base(backbone_lung, readout_head_lung, lung_dataset[0][0].shape, config_lung)

# Train the model for colon dataset
train_model(model_colon, colon_train_loader, colon_val_loader, max_epochs=10, accelerator='auto')

# Train the model for lung dataset
train_model(model_lung, lung_train_loader, lung_val_loader, max_epochs=10, accelerator='auto')

