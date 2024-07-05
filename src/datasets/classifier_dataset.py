import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
from PIL import Image


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
    def __init__(self, no_defects_folder: Path, defects_folder: Path, synthetized_defects_folder: Path = None):
        super().__init__()
        self.data = self.__load__(no_defects_folder, defects_folder, synthetized_defects_folder)

    def __len__(self):
        return len(self.data)

    def __load__(self, no_defects_folder: Path, defects_folder: Path, synthetized_defects_folder: Path = None) -> None:
        data = list()
        no_defect_images = list()
        defect_images = list()
        synthetized_defect_images = list()

        # Load no defect images paths
        no_defect_images = [os.path.join(no_defects_folder, image) 
                            for image in os.listdir(no_defects_folder) 
                            if os.path.isfile(os.path.join(no_defects_folder, image))]
        

        # Load defect images paths
        defect_images_folders = [os.path.join(defects_folder, folder) 
                                for folder in os.listdir(defects_folder) 
                                if os.path.isdir(os.path.join(defects_folder, folder))]
        for folder in defect_images_folders:
            images = os.listdir(folder)
            defect_image = min(images, key=len)
            defect_images.append(os.path.join(folder, defect_image))

        
        # Load synthetized defect images paths
        if synthetized_defects_folder:
            synthetized_defect_images =[os.path.join(synthetized_defects_folder, image) 
                                        for image in os.listdir(synthetized_defects_folder) 
                                        if os.path.isfile(os.path.join(synthetized_defects_folder, image))]

        for image_path in no_defect_images:
            data.append(
                {
                    'image_path': image_path,
                    'label': 0,
                }
            )

        for image_path in defect_images:
            data.append(
                {
                    'image_path': image_path,
                    'label': 1,
                }
            )

        if synthetized_defects_folder:
            for image_path in synthetized_defect_images:
                data.append(
                    {
                        'image_path': image_path,
                        'label': 1,
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
