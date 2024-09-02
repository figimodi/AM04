import os
import argparse
import random
from PIL import Image
import numpy as np
import albumentations as A
from glob import glob

def read_images_from_folder(folder_path):
    image_paths = glob(os.path.join(folder_path, '*.jpg'))  # Assuming images are in .jpg format
    images = [Image.open(image_path).convert('L') for image_path in image_paths]
    return images

def augment_image(image):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
        A.GaussNoise(p=0.5, var_limit=(10.0, 20.0), noise_scale_factor=.8),
    ])
    image_np = np.array(image)
    augmented_image_np = transform(image=image_np)['image']
    augmented_image = Image.fromarray(augmented_image_np).convert('RGB')
    return augmented_image

def generate_augmented_samples(input_folder, output_folder, n_samples):
    images = read_images_from_folder(input_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Remove old files in the output folder
    for file_name in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    for i in range(n_samples):
        augmented_image = augment_image(random.choice(images))
        output_path = os.path.join(output_folder, f'NoDefect_{i}.jpg')
        augmented_image.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate augmented images.')
    parser.add_argument('n_samples', type=int, help='Number of augmented samples to generate')
    args = parser.parse_args()
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_folder = os.path.join(data_path, 'NoDefects')
    output_folder = os.path.join(data_path, 'SyntheticNoDefects')
    generate_augmented_samples(input_folder, output_folder, args.n_samples)