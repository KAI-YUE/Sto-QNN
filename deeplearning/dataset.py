import os
import numpy as np
import pickle
import logging

# PyTorch Libraries
import torch
from torch.utils.data import Dataset

class CustomizedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = self._normalize(images)
        self.labels = (labels).astype(np.int64)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return dict(image=image, label=label)

    def _normalize(self, images):
        return images.astype(np.float32)/255
