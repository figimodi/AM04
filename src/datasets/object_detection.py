import os
import torch
import pickle
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from enum import Enum


class Defect(Enum):
    HOLE = 0
    VERTICAL = 1
    INCANDESCENCE = 2
    SPATTERING = 3
    HORIZONTAL = 4

class ObjectDetectionDatasetSplit(Dataset):
    def __init__(self, data: pd.DataFrame, annotations: Path):
        self.data = data
        self.mean, self.std = self.__calculate_mean_std__()
        
        with open(annotations, 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        sample = self.data.iloc[idx, :]
        image = Image.open(sample.image_path).convert('L')
        image = transform(image)
        targets = self.annotations[os.path.basename(sample.image_path)]

        return (image, targets)

    def __calculate_mean_std__(self):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        means = []
        stds = []
        
        for idx in range(len(self.data)):
            sample = self.data.iloc[idx, :]
            image = Image.open(sample.image_path).convert('L')
            image = transform(image)
            means.append(image.mean(dim=[1, 2]))
            stds.append(image.std(dim=[1, 2]))
            
        mean = torch.stack(means).mean(dim=0).numpy()
        std = torch.stack(stds).mean(dim=0).numpy()
        
        return mean, std

    def __iter__(self):
        for sample in self.data:
            yield sample

class ObjectDetectionDataset(Dataset):
    def __init__(self, synthetized_defects_folder):
        super().__init__()
        self.data = self.__load__(synthetized_defects_folder)

    def __len__(self):
        return len(self.data)

    def __load__(self, synthetized_defects_folder) -> None:
        data = list()
        synthetized_defect_images = list()
        
        # Load synthetized defect images paths
        synthetized_defect_images =[os.path.join(synthetized_defects_folder, image) 
                                    for image in os.listdir(synthetized_defects_folder) 
                                    if os.path.isfile(os.path.join(synthetized_defects_folder, image))]

        for image_path in synthetized_defect_images:
            data.append(
                {
                    'image_path': image_path,
                }
            )

        return pd.DataFrame(data)

    def create_splits(self, splits_proportion: List[float], annotations:Path) -> Tuple[ObjectDetectionDatasetSplit]:
        assert sum(splits_proportion) == 1, 'The proportions of the splits must be sum up to 1'

        # Calculate the number of rows for each split based on proportions
        split_sizes = [int(prop * len(self)) for prop in splits_proportion]
        
        # Shuffle the DataFrame
        df = self.data.sample(frac=1).reset_index(drop=True)
        
        # Split the DataFrame into chunks based on proportions
        splits = list()
        start_idx = 0
        for size in split_sizes:
            if size == 0:
                splits.append(None)
            else:
                splits.append(ObjectDetectionDatasetSplit(df.iloc[start_idx:start_idx + size], annotations))
                start_idx += size
    
        return tuple(splits)    

