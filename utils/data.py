from mimeta import MIMeta
from lightly.data import LightlyDataset
import torch
from torchvision import transforms
import yaml
from lightly.transforms.simclr_transform import SimCLRTransform

import os
from lightly.transforms.vicregl_transform import VICRegLTransform
from lightly.transforms.moco_transform import MoCoV2Transform

from utils.mimeta_warpper import MIMetaWrapper


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


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
    elif config['transform'] == 'VICRegLTransform':
        config['transform'] = VICRegLTransform(n_local_views=0)
    elif config['transform'] == 'MoCoTransform':
        config['transform'] = MoCoV2Transform(input_size = config['img_size'])
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

def calc_label_counts(dataloader):
    label_counts = {}
    with torch.no_grad():
        for _, targets in tqdm(dataloader, desc = 'Calculating Test Distribution'):
            counts = torch.unique(targets, return_counts = True)
            counts = torch.stack(counts, dim = 1)
            for cl, num in counts:
                if cl.item() not in label_counts.keys():
                    label_counts[cl.item()] = 0

                label_counts[cl.item()] += num.item()
        
        return label_counts

def confusion_matrix(targets, predictions, mode = 'percent'):

    num_classes = len(torch.unique(targets))

    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t,p in zip(targets, predictions):
            confusion_matrix[int(t.item()),int(p.item())] += 1
                
        if mode == 'percent':
            for i,div in enumerate(confusion_matrix.sum(axis = 1)):
                confusion_matrix[i] /= div
            confusion_matrix *= 100

        return confusion_matrix

def print_confusion_matrix(confusion_matrix):
    num_classes = len(confusion_matrix)
    torch.round(confusion_matrix, decimals = 2)
    horizontal = u'\u2500' * 7 * (num_classes + 1)
    print(f"{'cl':<6}\u2502", end = '')
    for i in range(num_classes):
        print(f'{i:<6}\u2502', end = '')
    print()
    print(horizontal)
    for i, row in enumerate(confusion_matrix):
        print(f'{i:<6}\u2502', end = '')
        for value in row:
            # print(value)
            value = round(value.item(),2)
            print(f'{value:<6}\u2502', end = '')
        print()
        print(horizontal)
    print()

def plot_confusion_matrix(confusion_matrix,display_labels = None, cmap = 'viridis'):
    """
    inspired by sklearn.metrics.ConfusionMatrixDisplay
    """
    fig, ax = plt.subplots()
    
    num_classes = len(confusion_matrix)

    if display_labels == None:
        display_labels = np.arange(num_classes)
    
    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    
    im = ax.imshow(confusion_matrix, **default_im_kw)

    cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)
    

    text = np.empty_like(confusion_matrix)

    threshold = (confusion_matrix.max() + confusion_matrix.min()) / 2

    for i in range(num_classes):
        for j in range(num_classes):
            color = cmap_min if confusion_matrix[i,j] > threshold else cmap_max
            default_text_kwargs = dict(ha="center", va="center", color = color)

            text_cell = str(round(confusion_matrix[i,j].item(), 2))
            ax.text(j, i, text_cell, **default_text_kwargs)


    ax.set( xticks = np.arange(num_classes),
            yticks = np.arange(num_classes),
            xticklabels = display_labels,
            yticklabels = display_labels,
            xlabel = 'Groundtruth',
            ylabel = 'Prediction')
    ax.set_ylim((num_classes - 0.5, -0.5))

    fig.colorbar(im, ax=ax, label = 'in %')
    
    plt.savefig('confusion_matrix.png')

        