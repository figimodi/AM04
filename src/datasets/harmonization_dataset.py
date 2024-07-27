import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from collections import defaultdict

class HarmonizationDatasetSplit(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        transform = transforms.ToTensor()

        sample = self.data.iloc[idx, :]
        original_image = transform(Image.open(sample.original_image).convert('L').resize((512, 512)))
        fake_image = transform(Image.open(sample.fake_image).convert('L').resize((512, 512)))
        mask_image = transform(Image.open(sample.mask_image).convert('L').resize((512, 512)))
        input_tensor = torch.cat((fake_image, mask_image), 0)
        return (original_image, input_tensor)

    def __iter__(self):
        for sample in self.data:
            yield sample

class HarmonizationDataset(Dataset):
    def __init__(self, defects_folder: Path, defects_masks_folder: Path):
        super().__init__()
        self.data = self.__load__(defects_folder, defects_masks_folder)

    def __len__(self):
        return len(self.data)

    def __load__(self, defects_folder: Path, defects_masks_folder: Path) -> pd.DataFrame:
        data = list()
        
        # Create the dictionary with the paths of masks {"ImageX" : [mask_path_K,...]}
        defect_masks = defaultdict(list)
        # Append the normal defect masks
        [defect_masks[img.split("_")[0]].append(os.path.join(defects_masks_folder, img)) for img in os.listdir(defects_masks_folder) if img.endswith(".jpg")]
        # Append the combined defect masks
        [defect_masks[img.split("_")[0]].append(os.path.join(defects_masks_folder, "combinations", img)) for img in os.listdir(os.path.join(defects_masks_folder, "combinations")) if img.endswith(".jpg")]

        for image_name in os.listdir(defects_folder):
            image_folder_path = os.path.join(defects_folder, image_name)
            if os.path.isdir(image_folder_path):
                images = os.listdir(image_folder_path)
                
                # Get image with the shortest name (original image)
                original_image = min(images, key=len)
                original_image_path = os.path.join(image_folder_path, original_image)

                # Remove the original image and leave the modified ones
                images.remove(original_image)
                
                # Associate each fake image with its corresponding mask
                for fake_image in images:
                    img_metadata = fake_image.split("_")
                    img_name = img_metadata[0]
                    img_mask = int(img_metadata[2])
                    
                    data.append(
                        {
                            'original_image': original_image_path,
                            'fake_image': os.path.join(image_folder_path, fake_image),
                            'mask_image': defect_masks[img_name][img_mask]
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
