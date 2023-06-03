from mimeta import MIMeta
from lightly.data import LightlyDataset
import torch
from torchvision import transforms
import yaml
from lightly.transforms.simclr_transform import SimCLRTransform

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
    config['transform'],])
    data = MIMeta(config['data']['path'], config['evaluation']['domain'], config['evaluation']['task'])

    litdata = LightlyDataset.from_torch_dataset(data, transform=make_rgb)

    data_splits = torch.utils.data.random_split(litdata, splits, generator = generator)

    return data_splits

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
