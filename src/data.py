from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode
        self.transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = Path('data') / self.data.iloc[idx, 0]
        image = imread(img_name)
        image = gray2rgb(image)
        image = self.transform(image)
        labels = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32)
        return image, labels

