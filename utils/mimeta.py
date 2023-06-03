import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .logger import logger
from torch.utils.data import Dataset 

def default_transform(input_size):
    return transforms.Compose(
        [
            transforms.Resize(input_size[1:]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * input_size[0], (0.5,) * input_size[0]),
        ]
    )


class MIMetaWrapper(Dataset):
    _data_path: str = None
    _infos: dict[str, dict] = None
    _dataset_dir_mapping: dict[str, str] = None
    _available_tasks: dict[str, list[str]] = None

    def __init__(
        self,
        data_path,
        datatask:list,
    ):
        self.data_path = data_path
        self.datatask = datatask
        
        

        # set number of channels based on input size
        self.num_channels = self.input_size[0]

        # set transform to default if none is provided
        if self.transform is None:
            self.transform = default_transform(self.input_size)

        # set up paths and load label data
        self.image_dirs = []
        self.label_files = []
        self._lookup = {}
        running_index = 0
        for dataset in datatask:
            dataset_subdir = self._dataset_dir_mapping[dataset[0]]
            image_dir = os.path.join(data_path, dataset_subdir, "images")
            label_file = os.path.join(
                data_path, dataset_subdir, "task_labels", f"{dataset[1]}.npy"
            )
            label = np.load(label_file)
            for i in range(len(label)):
                self._lookup[running_index] = os.path.join(image_dir, f"{i:06d}.tiff")
                running_index += 1
        self.len = running_index
        # creating look up tabel



        self.labels: np.ndarray = np.load(self.label_file)
        if self.task_target == TaskTarget.BINARY_CLASSIFICATION:
            self.labels = self.labels[:, np.newaxis]


    def __getitem__(self, index):
        image_path = self._lookup[index]
        image = Image.open(image_path)
        if image.mode == "RGB" and self.num_channels == 1:
            # convert RGB images to grayscale
            image = image.convert("L")
        elif image.mode == "L" and self.num_channels == 3:
            # convert grayscale images to RGB
            image = image.convert("RGB")
            # image = Image.merge("RGB", [image] * 3)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[0]
        return image, torch.as_tensor(label)

    def __len__(self):
        return self.len

    