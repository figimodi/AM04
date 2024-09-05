from enum import Enum
import os
import random
from PIL import Image
import numpy as np
import argparse
import pickle
from utils.guarantee_minimum_dimensions import guarantee_minimum_dimenions

import torch

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PATH_TO_MASKS = os.path.join(ROOT_DIR, 'data', 'DefectsMasks')
PATH_TO_DEFECTS = os.path.join(ROOT_DIR, 'data', 'Defects')
PATH_TO_NODEFECTS = os.path.join(ROOT_DIR, 'data', 'NoDefects')
PATH_TO_SYNTHETIC = os.path.join(ROOT_DIR, 'data', 'SyntheticDefects')
PATH_TO_SYNTHETIC_MASKS = os.path.join(ROOT_DIR, 'data', 'SyntheticDefectsMasks')
PATH_TO_SYNTHETIC_DEFECTS_HARMONIZED = os.path.join(ROOT_DIR, 'data', 'SyntheticDefectsHarmonized')
MIN_DIM_HOLE = 40

#borders of the frame. X and Y coordinates (origin is TOP LEFT). Taken using paint (while moving the cursor gives x and y coordinates)
FRAME_BORDERS = {
    'L': 240,
    'R': 1260,
    'T': 130,
    'B': 880
}

def delete_old_files():
    for fn in os.listdir(PATH_TO_SYNTHETIC):
        file_path = os.path.join(PATH_TO_SYNTHETIC, fn)
        os.remove(file_path)   
        
    for fn in os.listdir(PATH_TO_SYNTHETIC_MASKS):
        file_path = os.path.join(PATH_TO_SYNTHETIC_MASKS, fn)
        os.remove(file_path)        
                    
    print('Old synthetic images deleted!')

# Keeps a square that extactly contains true points of the mask
def crop_image(image):
    true_points = np.argwhere(image==255)  
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return cropped_arr

# returns the top left corrfinates of the mask (just the true points)
def get_pos_topleft(image):
    true_points = np.argwhere(image==255)  
    top_left = true_points.min(axis=0)
    return top_left

class Defect(Enum):
    HOLE = 0
    VERTICAL = 1
    INCANDESCENCE = 2
    SPATTERING = 3
    HORIZONTAL = 4
    
def main(samples_to_generate_per_defect = 10, probability_few_defects = .8):
    # probability of having few defects against having many defects (few and many defined next)
    masks_paths = [ os.path.join(PATH_TO_MASKS, mask_folder, mask_filename) for mask_folder in os.listdir(PATH_TO_MASKS) if os.path.isdir(os.path.join(PATH_TO_MASKS, mask_folder)) for mask_filename in os.listdir(os.path.join(PATH_TO_MASKS, mask_folder)) if (mask_filename.endswith('.jpg') or mask_filename.endswith('.png')) and '_PD_' in mask_filename and ((int(mask_folder.split('Image')[-1]) > 50 and 'Spattering' in mask_filename) or (int(mask_folder.split('Image')[-1]) < 50 and 'Spattering' not in mask_filename))]
    
    nodefects_filenames = [f for f in os.listdir(PATH_TO_NODEFECTS) if f.endswith('.jpg') or f.endswith('.png')]
    defect_types = list(set([f.split('_')[3].split('.')[0] for f in masks_paths]))
    data_faster_rcnn  = {}

    few = lambda: random.choice([1,2,3])
    many = lambda: random.choice([4,5,6])

    choose_how_many_defects = lambda: few() if random.random() < probability_few_defects else many()
    # choose_defect_type = lambda: random.choice(defect_types)
    get_defect_mask_path_by_type = lambda chosen_type: random.choice([p for p in masks_paths if (p.endswith('.jpg') or p.endswith('.png')) and p.split('_')[-1].split('.')[0] == chosen_type])
    
    print('Generating synthetic images...')
    
    print(f'0/{samples_to_generate_per_defect*len(defect_types)}', end='\r')
    
    for i_chosen_defect, chosen_defect_type in enumerate(defect_types):
        for i_samples_to_generate_per_defect in range(samples_to_generate_per_defect):
            nodefects_filename = random.choice(nodefects_filenames)
            path_to_nodefect_image = os.path.join(PATH_TO_NODEFECTS, nodefects_filename)
            nodefect_name = nodefects_filename.split('.')[0]
            
            nodefect_image = Image.open(path_to_nodefect_image).convert('RGB')
            
            number_of_defects = choose_how_many_defects() if 'Vertical' not in chosen_defect_type else random.randint(1,2)
            
            current_whole_mask = None
            
            synthetic_image_name = f'{nodefect_name}_{chosen_defect_type}_{i_samples_to_generate_per_defect}.jpg'
            
            data_faster_rcnn_item = {
                "boxes":  [],   #[x_min, y_min, x_max, y_max]
                "labels": []    #defect type
            }
            
            for i_number_of_defects in range(number_of_defects):
                if i_number_of_defects == 0:
                    current_whole_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                
                defect_mask_path = get_defect_mask_path_by_type(chosen_defect_type)
                defect_name =  os.path.basename(defect_mask_path).split('_')[0]
                path_to_defect_image = os.path.join(PATH_TO_DEFECTS, defect_name, f'{defect_name}.jpg')
                
                defect_image = Image.open(path_to_defect_image).convert('RGB')
                defect_mask = Image.open(defect_mask_path).convert('L')
                
                if os.path.basename(defect_mask_path) == 'Image43_PD_05_Vertical.png':
                    defect_image = defect_image.rotate(-3)
                    defect_mask = defect_mask.rotate(-3)
                    original_side_is_right = True
                    
                if os.path.basename(defect_mask_path) == 'Image34_PD_02_Vertical.png':
                    defect_image = defect_image.rotate(.5)
                    defect_mask = defect_mask.rotate(.5)
                    original_side_is_right = False
                    
                #get the top left coordinates of the mask
                original_topleft_pos = get_pos_topleft(np.array(defect_mask))
                proto_cropped_mask = crop_image(np.array(defect_mask))
                
                count_tries = 0
                while count_tries < 10:

                    x_start = random.randint(FRAME_BORDERS['L'], FRAME_BORDERS['R'] - proto_cropped_mask.shape[1])
                    y_start = random.randint(FRAME_BORDERS['T'], FRAME_BORDERS['B'] - proto_cropped_mask.shape[0])
                    
                    test_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                    
                    x_end = x_start + proto_cropped_mask.shape[1]
                    y_end = y_start + proto_cropped_mask.shape[0]
                    
                    current_side_is_right = False if (x_start+x_end)//2 < 760 else True
                
                    if 'Vertical' in chosen_defect_type and original_side_is_right != current_side_is_right:
                        defect_mask = Image.fromarray(np.fliplr(defect_mask))
                        defect_image = Image.fromarray(np.fliplr(defect_image))
                        
                        if current_side_is_right:
                            x_start = x_start - 20
                        else:
                            x_start = x_start + 20
                        
                        original_topleft_pos = get_pos_topleft(np.array(defect_mask))
                        flipped_mask = crop_image(np.array(defect_mask))
                        x_end = x_start + flipped_mask.shape[1]
                        y_end = y_start + flipped_mask.shape[0]
                        
                        test_mask[y_start:y_end, x_start:x_end] = flipped_mask
                    else:     
                        test_mask[y_start:y_end, x_start:x_end] = proto_cropped_mask
                    
                    if not np.any(test_mask*current_whole_mask) or i_number_of_defects==0:
                        current_whole_mask = (current_whole_mask > 127).astype(np.uint8) * 255
                        test_mask = (test_mask > 127).astype(np.uint8) * 255
                        current_whole_mask = np.logical_or(current_whole_mask, test_mask).astype(np.uint8) * 255
                        count_tries = 11
                    else:
                        count_tries += 1
                
                if Defect[chosen_defect_type.upper()].value == Defect.HOLE.value:
                    x_start, y_start, x_end, y_end = guarantee_minimum_dimenions(x_start, y_start, x_end, y_end, min_dim_w=MIN_DIM_HOLE, min_dim_h=MIN_DIM_HOLE)
                
                data_faster_rcnn_item["boxes"].append([x_start * 512 / 1280, y_start * 512 / 1024, x_end * 512 / 1280, y_end * 512 / 1024])

                data_faster_rcnn_item["labels"].append()
                
                nodefect_image.paste(defect_image, (x_start - original_topleft_pos[1], y_start - original_topleft_pos[0]), defect_mask)
            
            data_faster_rcnn_item["boxes"] = torch.tensor(data_faster_rcnn_item["boxes"]).to(torch.float32)
            data_faster_rcnn_item["labels"] = torch.tensor(data_faster_rcnn_item["labels"]).to(torch.int64)
            
            data_faster_rcnn[synthetic_image_name] = data_faster_rcnn_item    
            
            # save
            nodefect_image.save(os.path.join(PATH_TO_SYNTHETIC, synthetic_image_name))
            Image.fromarray(current_whole_mask).convert('L').save(os.path.join(PATH_TO_SYNTHETIC_MASKS, f'{nodefect_name}_{chosen_defect_type}_{i_samples_to_generate_per_defect}_mask.png'))   
            print(f'{i_samples_to_generate_per_defect + i_chosen_defect*samples_to_generate_per_defect + 1}/{samples_to_generate_per_defect*len(defect_types)}', end='\r')
            
    pickle.dump(data_faster_rcnn, open(os.path.join(PATH_TO_SYNTHETIC, '..', 'data_faster_rcnn.pkl'), 'wb'))
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--tot_samples', type=int, help='Amount of synthetic images generated', dest='N_SAMPLES', default=500)    
    args = parser.parse_args()

    os.makedirs(PATH_TO_SYNTHETIC, exist_ok=True)
    os.makedirs(PATH_TO_SYNTHETIC_MASKS, exist_ok=True)
    os.makedirs(PATH_TO_SYNTHETIC_DEFECTS_HARMONIZED, exist_ok=True)
    delete_old_files()
    main(samples_to_generate_per_defect=args.N_SAMPLES//5)
    