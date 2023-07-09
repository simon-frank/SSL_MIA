from mimeta import MIMeta
from lightly.data import LightlyDataset
import torch
from torchvision import transforms
import yaml
from lightly.transforms.simclr_transform import SimCLRTransform
import os
from utils.mimeta_warpper import MIMetaWrapper


"""
Helper function to get the data splits always with the same seed and adjusted wrapper for the pretraining
"""
def get_data_pretraining(config):

    manuel_seed = 42 #do not change

    splits = [0.8,0.1,0.1]

    generator = torch.Generator().manual_seed(manuel_seed)

    make_rgb = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    config['transform'],])
    alldatasets=[]

    for dataset in config['data']['datasets']:
        alldatasets.append([dataset['domain'], dataset['task']])

    data = MIMetaWrapper(config['data']['path'], alldatasets)

    litdata = LightlyDataset.from_torch_dataset(data, transform=make_rgb)

    data_splits = torch.utils.data.random_split(litdata, splits, generator = generator)

    return data_splits

"""
Helper function to get the data splits always with the same seed and adjusted wrapper for the pretraining
"""
# TODO: use mimeta
def get_data_finetuning(config):

    manuel_seed = 42 #do not change

    splits = [0.8,0.1,0.1]

    generator = torch.Generator().manual_seed(manuel_seed)

    make_rgb = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),])
    data = MIMeta(config['data']['path'], config['evaluation']['domain'], config['evaluation']['task'], transform=make_rgb)

    #litdata = LightlyDataset.from_torch_dataset(data, transform=make_rgb)

    train, val, test = torch.utils.data.random_split(data, splits, generator = generator)

    traindata = config['finetuning']['trainsplit']
    train_splits =[traindata, 1-traindata ]
    train,_ = torch.utils.data.random_split(train, train_splits, generator = generator)
    return train, val, test

"""
Helper function to load the config file
"""

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['optimizer'] == 'SGD':
        config['optimizer'] = torch.optim.SGD
    if config['optimizer'] == 'Adam':
        config['optimizer'] = torch.optim.Adam
    if config['transform'] == 'SimCLRTransform':
        config['transform'] = SimCLRTransform(input_size = config['img_size'])

    return config

def save_config(data, path):
    output_file = os.path.join(path, 'config.yaml')
    with open(output_file, "w") as file:
        yaml.dump(data, file)

def save_performance(test_loss, test_accuracy, save_directory):
    
    # Specify the filename for the output text file
    output_file = output_file = os.path.join(save_directory, "results.txt")
    
    # Save the test loss and accuracy to the file
    with open(output_file, "w") as file:
        file.write(f"Test Loss: {test_loss}\n")
        file.write(f"Test Accuracy: {test_accuracy}\n")
    
"""
Helper function to calculate the balance in the dataset
"""


def calculate_label_counts(dataset):
    label_counts = torch.zeros(11)

    for _, label in dataset:
        label_counts[label] += 1

    return label_counts
