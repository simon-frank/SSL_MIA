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
    

    @classmethod
    def get_available_datasets(cls, data_path: str) -> list[str]:
        if cls._dataset_dir_mapping is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        return list(cls._dataset_dir_mapping.keys())

    @classmethod
    def get_available_tasks(cls, data_path: str) -> dict[str, list[str]]:
        if cls._available_tasks is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        return cls._available_tasks

    @classmethod
    def get_info_dict(cls, data_path: str, dataset_name: str) -> dict:
        if cls._dataset_dir_mapping is None or cls._data_path != data_path:
            cls._read_dataset_info(data_path)
        if dataset_name not in cls._infos:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {list(cls._infos.keys())}"
            )
        return cls._infos[dataset_name]

    @classmethod
    def _read_dataset_info(cls, data_path: str):
        cls._dataset_dir_mapping = {}
        cls._available_tasks = {}
        cls._infos = {}
        for subdir in os.listdir(data_path):
            subdir_path = os.path.join(data_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            info_path = os.path.join(subdir_path, "info.yaml")
            if not os.path.isfile(info_path):
                continue
            with open(info_path, "r") as f:
                info = yaml.load(f, Loader=yaml.FullLoader)
                cls._infos[info["name"]] = info
                cls._dataset_dir_mapping[info["name"]] = subdir
                cls._available_tasks[info["name"]] = [
                    t["task_name"] for t in info["tasks"]
                ]
        cls._data_path = data_path


    