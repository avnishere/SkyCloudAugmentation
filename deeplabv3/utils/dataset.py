import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from logger import logger


class SkyDataset(Dataset):
    """
    A PyTorch Dataset class for loading and transforming images and labels for sky segmentation.

    Attributes:
        image_paths (list): List of paths to the image files.
        label_paths (list): List of paths to the label files.
        train_transform (callable, optional): A function/transform to apply to the training images.
        base_transform (callable): A function/transform to apply to all images.
        to_tensor (callable): A function/transform to convert images to PyTorch tensors and normalize them.
        mapping (dict): A mapping from original label values to new class values.
    """

    def __init__(
        self,
        image_path: str,
        label_path: str,
        train_transform: Optional[Callable] = None,
        base_transform: Optional[Callable] = None,
    ):
        """
        Initializes the SkyDataset with the paths to the images and labels, and the transforms to apply.

        Args:
            image_path (str): Path to the directory containing the image files.
            label_path (str): Path to the directory containing the label files.
            train_transform (callable, optional): A function/transform to apply to the training images.
            base_transform (callable, optional): A function/transform to apply to all images.
        """
        self.image_paths = [image_path + image for image in os.listdir(image_path)]
        self.label_paths = [label_path + label for label in os.listdir(label_path)]
        self.train_transform = train_transform
        self.base_transform = base_transform
        self.to_tensor = Compose([ToTensor(), Normalize(0.5, 0.5)])
        self.mapping = {
            0: 0,
            255: 1,
            128: 2,
        }
        logger.info(f"Loading data from {image_path}. Dataset size: {self.__len__()}")

    def __getitem__(self, index: int):
        """
        Returns the image and label at the specified index after applying the necessary transforms.

        Args:
            index (int): Index of the image and label to return.

        Returns:
            tuple: The transformed image and label.
        """
        # Load image and label
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.base_transform:
            image = self.base_transform(image)
            label = self.base_transform(label)

        if self.train_transform:
            image, label = self.train_transform(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.uint8))
        label = self.mask_to_class(label)

        return self.to_tensor(image), label.long()

    def __len__(self):
        """Returns the number of images/labels in the dataset."""
        return len(self.image_paths)

    def mask_to_class(self, label: torch.Tensor):
        """
        Converts the original label values to new class values based on the mapping.

        Args:
            label (torch.Tensor): The original label.

        Returns:
            torch.Tensor: The label with new class values.
        """
        for k in self.mapping:
            label[label == k] = self.mapping[k]
        return label
