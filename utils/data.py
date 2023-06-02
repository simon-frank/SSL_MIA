from mimeta import MIMeta
from lightly.data import LightlyDataset
import torch
from torchvision import transforms

"""
Helper function to get the data splits always with the same seed
"""
def get_data(config):

    manuel_seed = 42

    splits = [0.8,0.1,0.1]

    generator = torch.Generator().manual_seed(manuel_seed)

    make_rgb = transforms.Grayscale(num_output_channels = 3)

    alldatasets=[]

    for dataset in config['data']['datasets']:
        alldatasets.append(MIMeta(config['data']['path'], dataset['domain'], dataset['task'], transform = make_rgb))


    data = torch.utils.data.ConcatDataset(alldatasets)

    litdata = LightlyDataset.from_torch_dataset(data, transform=config['transform'])

    data_splits = torch.utils.data.random_split(litdata, splits, generator = generator)

    return data_splits

