import os
import pandas as pd
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

class ClassifierDatasetSplit(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        transform = transforms.ToTensor()

        sample = self.data.iloc[idx, :]
        image = transform(Image.open(sample.image_path).convert('RGB').resize((256, 256)))
        label = sample.label
        return (image, label)

    def __iter__(self):
        for sample in self.data:
            yield sample

class ClassifierDataset(Dataset):
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
                    'label': Defect[os.path.splitext(image_path)[0].split('_')[-1].upper()].value,
                }
            )

        return pd.DataFrame(data)

    def create_splits(self, splits_proportion: List[float]) -> Tuple[ClassifierDatasetSplit]:
        assert sum(splits_proportion) == 1, 'The proportions of the splits must be sum up to 1'

        # Calculate the number of rows for each split based on proportions
        split_sizes = [int(prop * len(self)) for prop in splits_proportion]
        
        # Shuffle the DataFrame
        df = self.data.sample(frac=1).reset_index(drop=True)
        
        # Split the DataFrame into chunks based on proportions
        splits = list()
        start_idx = 0
        for size in split_sizes:
            splits.append(ClassifierDatasetSplit(df.iloc[start_idx:start_idx + size]))
            start_idx += size
    
        return tuple(splits)    
