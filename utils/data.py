from mimeta import MIMeta
from lightly.data import LightlyDataset
import torch
from torchvision import transforms
import yaml
from lightly.transforms.simclr_transform import SimCLRTransform

"""
Helper function to get the data splits always with the same seed
"""
def get_data(config):

    manuel_seed = 42

    splits = [0.8,0.1,0.1]

    generator = torch.Generator().manual_seed(manuel_seed)

    make_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Other transformations...
])
    
    #transforms.Grayscale(num_output_channels = 3)

    alldatasets=[]

    for dataset in config['data']['datasets']:
        alldatasets.append(MIMeta(config['data']['path'], dataset['domain'], dataset['task'], transform = make_rgb))


    #data = torch.utils.data.ConcatDataset(alldatasets)

    litdata = LightlyDataset.from_torch_dataset(alldatasets[0], transform=config['transform'])

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
