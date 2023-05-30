from mimeta import MIMeta
from lightly.data import LightlyDataset
import torch

def get_data(config):

    generator = torch.Generator().manual_seed(config['seed'])
    data = MIMeta(config['data'], config['domain'], config['task'])
    litdata = LightlyDataset.from_torch_dataset(data, transform=config['transform'])

    data_splits = torch.utils.data.random_split(litdata, config['splits'], generator = generator)

    return data_splits

