import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
from PIL import Image


class HarmonizationDatasetSplit(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        transform = transforms.ToTensor()

        sample = self.data.iloc[idx, :]
        original_image = transform(Image.open(sample.original_image).convert('RGB').resize((256, 256)))
        fake_image = transform(Image.open(sample.fake_image).convert('RGB').resize((256, 256)))
        return (original_image, fake_image)

    def __iter__(self):
        for sample in self.data:
            yield sample

class HarmonizationDataset(Dataset):
    def __init__(self, defects_folder: Path):
        super().__init__()
        self.data = self.__load__(defects_folder)

    def __len__(self):
        return len(self.data)

    def __load__(self, defects_folder: Path) -> None:
        data = list()

        for image_name in os.listdir(defects_folder):
            image_folder_path = os.path.join(defects_folder, image_name)
            if os.path.isdir(image_folder_path):
                images = os.listdir(image_folder_path)
                
                # Get image with the shortest name
                original_image = min(images, key=len)
                original_image_path = os.path.join(image_folder_path, original_image)

                # Get list of fake images
                images.remove(original_image)
                fake_images_path = [os.path.join(image_folder_path, fake_image) for fake_image in images]

                for fake_image_path in fake_images_path:
                    data.append(
                        {
                            'original_image': original_image_path,
                            'fake_image': fake_image_path,
                        }
                    )

        return pd.DataFrame(data)

    def create_splits(self, splits_proportion: List[float]) -> Tuple[HarmonizationDatasetSplit]:
        assert sum(splits_proportion) == 1, 'The proportions of the splits must be sum up to 1'

        # Calculate the number of rows for each split based on proportions
        split_sizes = [int(prop * len(self)) for prop in splits_proportion]
        
        # Shuffle the DataFrame
        df = self.data.sample(frac=1).reset_index(drop=True)
        
        # Split the DataFrame into chunks based on proportions
        splits = list()
        start_idx = 0
        for size in split_sizes:
            splits.append(HarmonizationDatasetSplit(df.iloc[start_idx:start_idx + size]))
            start_idx += size
    
        return tuple(splits)    
