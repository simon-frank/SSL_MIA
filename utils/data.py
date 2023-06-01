from mimeta import MIMeta
from lightly.data import LightlyDataset
import torch
from torchvision import transforms

def get_data(config):

    generator = torch.Generator().manual_seed(config['seed'])

    make_rgb = transforms.Grayscale(num_output_channels = 3)

    data = MIMeta(config['data'], config['domain'], config['task'], transform = make_rgb)

    litdata = LightlyDataset.from_torch_dataset(data, transform=config['transform'])

    data_splits = torch.utils.data.random_split(litdata, config['splits'], generator = generator)

    return data_splits

