import numpy as np
import argparse
import shutil
import random
import os
from utils.spattering import generate_images_with_random_proliferation
from PIL import Image


def clear_old_spattering_generations(end_true_masks, defects_masks_path, defects_paths):
    for folder_name in os.listdir(defects_masks_path):
        folder_path = os.path.join(defects_masks_path, folder_name)
        index = int(folder_name.split('Image')[-1])
        if os.path.isdir(folder_path) and index > end_true_masks:
            shutil.rmtree(folder_path)
            
    for folder_name in os.listdir(defects_paths):
        folder_path = os.path.join(defects_paths, folder_name)
        index = int(folder_name.split('Image')[-1])
        if os.path.isdir(folder_path) and index > end_true_masks:
            shutil.rmtree(folder_path)
            
    print("Old spattering generations deleted!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--min_dist', type=int, help='Minimum distance between points', dest='MIN_DIST', default=0)    
    parser.add_argument('--max_points', type=int, help='Max number of points in the area', dest='MAX_POINTS', default=400)    
    parser.add_argument('--max_spread', type=int, help='Maximum spread among points', dest='MAX_SPREAD', default=10)    
    parser.add_argument('--darkest_gray', type=int, help='The hexadecimal value for the darkest gray', dest='DARKEST_GRAY', default=160)    
    parser.add_argument('--lightest_gray', type=int, help='The hexadecimal value for the lightest gray', dest='LIGHTEST_GRAY', default=170)    
    args = parser.parse_args()

    BASE_PATH = '../data'  # Replace with your base directory path
    DEFECTS_PATH = os.path.join(BASE_PATH, 'Defects')
    NODEFECTS_PATH = os.path.join(BASE_PATH, 'NoDefects')
    DEFECTS_MASKS_PATH = os.path.join(BASE_PATH, 'DefectsMasks')
    
    N_SPATTERING_IMAGES = 10
    START_SPATTERING_INDEX = 50
    END_TRUE_MASK_INDEX = 47
    
    clear_old_spattering_generations(END_TRUE_MASK_INDEX, DEFECTS_MASKS_PATH, DEFECTS_PATH)

    list_spattering_regions_paths = [ os.path.join(DEFECTS_MASKS_PATH, folder_name, image_name) for folder_name in os.listdir(DEFECTS_MASKS_PATH) for image_name in os.listdir(os.path.join(DEFECTS_MASKS_PATH, folder_name)) if os.path.isdir(os.path.join(DEFECTS_MASKS_PATH, folder_name)) and image_name.endswith('.png') and '_Spattering' in image_name ]
    list_nodefects_paths = [ os.path.join(NODEFECTS_PATH, image_name) for image_name in os.listdir(NODEFECTS_PATH) if image_name.endswith('.jpg') ]    
    
    print("Generating spattering images...")
    print(f'0/{N_SPATTERING_IMAGES}', end='\r')
    
    for i in range(N_SPATTERING_IMAGES):
        current_nodefect_path = random.choice(list_nodefects_paths)
        current_spattering_region_path = random.choice(list_spattering_regions_paths)
        
        current_nodefect = Image.open(current_nodefect_path).convert('RGB')
        current_spattering_region = np.array(Image.open(current_spattering_region_path).convert('L'))
        
        spattering, spattering_mask = generate_images_with_random_proliferation(
            current_spattering_region, min_dist=args.MIN_DIST, max_points=args.MAX_POINTS, max_spread=args.MAX_SPREAD, darkest_gray=args.DARKEST_GRAY, lightest_gray=args.LIGHTEST_GRAY
        )
        
        spattering = Image.fromarray(spattering).convert('RGB')
        spattering_mask = Image.fromarray(spattering_mask).convert('L')
        current_nodefect.paste(spattering, (0, 0), spattering_mask) 
        
        #save grountruth spattering image
        newdefect_spattering_foldername=f'Image{START_SPATTERING_INDEX+i}'
        newdefect_spattering_filename=f'{newdefect_spattering_foldername}.jpg'
        os.makedirs(os.path.join(DEFECTS_PATH, newdefect_spattering_foldername), exist_ok=True)
        newdefect_spattering_path = os.path.join(DEFECTS_PATH, newdefect_spattering_foldername)
        current_nodefect.save(os.path.join(newdefect_spattering_path, newdefect_spattering_filename))
        
        #save spattering masks
        newdefect_mask_foldername = newdefect_spattering_foldername
        newdefect_mask_filename = f'{newdefect_mask_foldername}_PD_01_Spattering.png'
        newdefect_mask_union_filename = f'{newdefect_mask_foldername}_Mask_01.png'
        os.makedirs(os.path.join(DEFECTS_MASKS_PATH, newdefect_mask_foldername), exist_ok=True)
        spattering_mask.save(os.path.join(DEFECTS_MASKS_PATH, newdefect_mask_foldername, newdefect_mask_filename))
        spattering_mask.save(os.path.join(DEFECTS_MASKS_PATH, newdefect_mask_foldername, newdefect_mask_union_filename))
        
        print(f'{i+1}/{N_SPATTERING_IMAGES}', end='\r')
        
    print("Done!")
    