import os
import numpy as np
import pickle
import logging

# PyTorch Libraries
import torch
from torch.utils.data import Dataset

class CustomizedDataset(Dataset):
    def __init__(self, images, labels, type_):
        self.images = self._normalize(images, type_)
        self.labels = (labels).astype(np.int64)
        self.images = torch.from_numpy(self.images)
        self.labels = torch.from_numpy(self.labels)

        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return dict(image=image, label=label)

    def _normalize(self, images, type_):
        if type_ == "mnist":
            images = images.astype(np.float32)/255
            images = (images - 0.1307)/0.3081
        elif type_ == "fmnist":
            images = images.astype(np.float32)/255
            images = (images - 0.2860)/0.3530
        elif type_ == "cifar":
            image_area = 32**2
            images = images.astype(np.float32)/255
            images[:, :image_area] = (images[:, :image_area] - 0.4914) / 0.247                              # r channel 
            images[:, image_area:2*image_area] = (images[:, image_area:2*image_area] - 0.4822) / 0.243      # g channel
            images[:, -image_area:] = (images[:, -image_area:] - 0.4465) / 0.261                            # b channel
        else:
            images = images.astype(np.float32)/255

        return images    
