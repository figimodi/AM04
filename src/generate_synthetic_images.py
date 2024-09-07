from enum import Enum
import os
import random
from PIL import Image
import numpy as np
import argparse
import pickle
from utils.utils import guarantee_minimum_dimenions

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

class Defect(Enum):
    HOLE = 0
    VERTICAL = 1
    INCANDESCENCE = 2
    SPATTERING = 3
    HORIZONTAL = 4
    
few = lambda: random.choice([1,2,3])
many = lambda: random.choice([4,5,6])

masks_paths = [ os.path.join(PATH_TO_MASKS, mask_folder, mask_filename) for mask_folder in os.listdir(PATH_TO_MASKS) if os.path.isdir(os.path.join(PATH_TO_MASKS, mask_folder)) for mask_filename in os.listdir(os.path.join(PATH_TO_MASKS, mask_folder)) if (mask_filename.endswith('.jpg') or mask_filename.endswith('.png')) and '_PD_' in mask_filename and ((int(mask_folder.split('Image')[-1]) > 50 and 'Spattering' in mask_filename) or (int(mask_folder.split('Image')[-1]) < 50 and 'Spattering' not in mask_filename))]

nodefects_filenames = [f for f in os.listdir(PATH_TO_NODEFECTS) if f.endswith('.jpg') ]
defect_types = [d.name.title() for d in Defect]
data_faster_rcnn  = {}

choose_how_many_defects = lambda probability_few_defects = .8: few() if random.random() < probability_few_defects else many()
get_defect_mask_path_by_type = lambda chosen_type: random.choice([p for p in masks_paths if p.endswith('.png') and p.split('_')[-1].split('.')[0] == chosen_type])

def delete_old_files():
    for fn in os.listdir(PATH_TO_SYNTHETIC):
        file_path = os.path.join(PATH_TO_SYNTHETIC, fn)
        os.remove(file_path)   
        
    for fn in os.listdir(PATH_TO_SYNTHETIC_MASKS):
        file_path = os.path.join(PATH_TO_SYNTHETIC_MASKS, fn)
        os.remove(file_path)        
                    
    print('Old synthetic images deleted!')
def crop_image(image):
    true_points = np.argwhere(image==255)  
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return cropped_arr
def get_pos_topleft(image):
    true_points = np.argwhere(image==255)  
    top_left = true_points.min(axis=0)
    return top_left

def single_defect_type_generation(num_tot_samples):
    
    print('Generating single defect type synthetic images...')
    
    print(f'0/{num_tot_samples}', end='\r')
    
    for i_chosen_defect, chosen_defect_type in enumerate(defect_types):
        for i_th_generation in range(num_tot_samples//len(defect_types)):
            data_faster_rcnn_item = {
                "boxes":  [],   #[x_min, y_min, x_max, y_max]
                "labels": []    #defect type
            }
            
            nodefects_filename = random.choice(nodefects_filenames)
            path_to_nodefect_image = os.path.join(PATH_TO_NODEFECTS, nodefects_filename)
            nodefect_name = nodefects_filename.split('.')[0]
            
            nodefect_image = Image.open(path_to_nodefect_image).convert('RGB')
            
            number_of_defects = choose_how_many_defects() if 'Vertical' not in chosen_defect_type else random.randint(1,2)
            
            current_whole_mask = None
            
            synthetic_image_name = f'{nodefect_name}_{chosen_defect_type}_{i_th_generation}.jpg'
            
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
                cropped_mask = crop_image(np.array(defect_mask))
                
                count_tries = 0
                while count_tries < 10:

                    x_start = random.randint(FRAME_BORDERS['L'], FRAME_BORDERS['R'] - cropped_mask.shape[1])
                    y_start = random.randint(FRAME_BORDERS['T'], FRAME_BORDERS['B'] - cropped_mask.shape[0])
                    
                    test_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                    
                    x_end = x_start + cropped_mask.shape[1]
                    y_end = y_start + cropped_mask.shape[0]
                    
                    current_side_is_right = False if (x_start+x_end)//2 < 760 else True
                
                    if 'Vertical' in chosen_defect_type and original_side_is_right != current_side_is_right:
                        defect_mask = Image.fromarray(np.fliplr(defect_mask))
                        defect_image = Image.fromarray(np.fliplr(defect_image))
                        
                        if current_side_is_right:
                            x_start = x_start - 20
                        else:
                            x_start = x_start + 20
                        
                        original_topleft_pos = get_pos_topleft(np.array(defect_mask))
                        flipped_cropped_mask = crop_image(np.array(defect_mask))
                        x_end = x_start + flipped_cropped_mask.shape[1]
                        y_end = y_start + flipped_cropped_mask.shape[0]
                        
                        test_mask[y_start:y_end, x_start:x_end] = flipped_cropped_mask
                    else:     
                        test_mask[y_start:y_end, x_start:x_end] = cropped_mask
                    
                    if not np.any(test_mask*current_whole_mask) or i_number_of_defects==0:
                        current_whole_mask = (current_whole_mask > 127).astype(np.uint8) * 255
                        test_mask = (test_mask > 127).astype(np.uint8) * 255
                        current_whole_mask = np.logical_or(current_whole_mask, test_mask).astype(np.uint8) * 255
                        count_tries = 11
                    else:
                        count_tries += 1
                        
                topLeft_bottomRight = [x_start * 512 / 1280, y_start * 512 / 1024, x_end * 512 / 1280, y_end * 512 / 1024]
                
                if Defect[chosen_defect_type.upper()].value == Defect.HOLE.value:
                    topLeft_bottomRight = guarantee_minimum_dimenions(*topLeft_bottomRight, min_dim_w=MIN_DIM_HOLE, min_dim_h=MIN_DIM_HOLE)
                
                data_faster_rcnn_item["boxes"].append(topLeft_bottomRight)

                data_faster_rcnn_item["labels"].append(Defect[chosen_defect_type.upper()].value)
                
                nodefect_image.paste(defect_image, (x_start - original_topleft_pos[1], y_start - original_topleft_pos[0]), defect_mask)
            
            data_faster_rcnn_item["boxes"] = torch.tensor(data_faster_rcnn_item["boxes"]).to(torch.float32)
            data_faster_rcnn_item["labels"] = torch.tensor(data_faster_rcnn_item["labels"]).to(torch.int64)
            
            data_faster_rcnn[synthetic_image_name] = data_faster_rcnn_item    
            
            # save
            nodefect_image.save(os.path.join(PATH_TO_SYNTHETIC, synthetic_image_name))
            Image.fromarray(current_whole_mask).convert('L').save(os.path.join(PATH_TO_SYNTHETIC_MASKS, f'{nodefect_name}_{chosen_defect_type}_{i_th_generation}_mask.png'))   
            print(f'{i_th_generation + i_chosen_defect*num_tot_samples//len(defect_types) + 1}/{num_tot_samples}', end='\r')
            
    pickle.dump(data_faster_rcnn, open(os.path.join(PATH_TO_SYNTHETIC, '..', 'data_faster_rcnn.pkl'), 'wb'))
    print('Done!')

def multi_defect_type_generation(num_tot_samples):
    print('Generating multi defect type synthetic images...')
    
    print(f'0/{num_tot_samples}', end='\r')
    
    for i_th_sample in range(num_tot_samples):
        number_of_defects = choose_how_many_defects()
        
        nodefects_filename = random.choice(nodefects_filenames)
        path_to_nodefect_image = os.path.join(PATH_TO_NODEFECTS, nodefects_filename)
        nodefect_name = nodefects_filename.split('.')[0]
        
        nodefect_image = Image.open(path_to_nodefect_image).convert('RGB')
        
        current_whole_mask = None
        
        synthetic_image_name = f'{nodefect_name}_MDT_{i_th_sample}.jpg'
        
        data_faster_rcnn_item = { "boxes":  [], "labels": [] }
        
        if random.random() > .95:  
            number_of_defects = 15
            
        for i_th_defect in range(number_of_defects):
            chosen_defect_type = random.choice(defect_types) if number_of_defects < 10 else Defect.INCANDESCENCE.name.title()
            
            if i_th_defect == 0:
                current_whole_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
            
            defect_mask_path = get_defect_mask_path_by_type(chosen_defect_type)
            defect_name =  os.path.basename(defect_mask_path).split('_')[0]
            path_to_defect_image = os.path.join(PATH_TO_DEFECTS, defect_name, f'{defect_name}.jpg')
            
            defect_image = Image.open(path_to_defect_image).convert('RGB')
            defect_mask = Image.open(defect_mask_path).convert('L')
            
            if os.path.basename(defect_mask_path) == 'Image43_PD_05_Vertical.png':
                defect_image = defect_image.rotate(-1.7)
                defect_mask = defect_mask.rotate(-1.7)
                original_side_is_right = True
                
            if os.path.basename(defect_mask_path) == 'Image34_PD_02_Vertical.png':
                defect_image = defect_image.rotate(.5)
                defect_mask = defect_mask.rotate(.5)
                original_side_is_right = False
                
            #get the top left coordinates of the mask
            original_topleft_pos = get_pos_topleft(np.array(defect_mask))
            cropped_mask = crop_image(np.array(defect_mask))
            
            count_tries = 0
            while count_tries < 10:
                
                if chosen_defect_type == Defect.VERTICAL.name.title():
                    x_start = random.randint(FRAME_BORDERS['L'] + 10, FRAME_BORDERS['L'] + 60) if random.randint(1,2)%2==0 else random.randint(FRAME_BORDERS['R'] - 80, FRAME_BORDERS['R'] - 50)
                    y_start = random.randint(FRAME_BORDERS['T'], FRAME_BORDERS['B'] - cropped_mask.shape[0])
                else:
                    x_start = random.randint(FRAME_BORDERS['L'], FRAME_BORDERS['R'] - cropped_mask.shape[1])
                    y_start = random.randint(FRAME_BORDERS['T'], FRAME_BORDERS['B'] - cropped_mask.shape[0])
                
                test_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                
                x_end = x_start + cropped_mask.shape[1]
                y_end = y_start + cropped_mask.shape[0]
                
                current_side_is_right = True if (x_start+x_end)//2 >= 760 else False
            
                if chosen_defect_type == Defect.VERTICAL.name.title() and original_side_is_right != current_side_is_right:
                    defect_mask = Image.fromarray(np.fliplr(defect_mask))
                    defect_image = Image.fromarray(np.fliplr(defect_image))
                    
                    original_topleft_pos = get_pos_topleft(np.array(defect_mask))
                    cropped_mask = crop_image(np.array(defect_mask))
                    x_end = x_start + cropped_mask.shape[1]
                    y_end = y_start + cropped_mask.shape[0]
                
                test_mask[y_start:y_end, x_start:x_end] = cropped_mask
                
                if not np.any(test_mask*current_whole_mask) or i_th_defect==0:
                    current_whole_mask = (current_whole_mask > 127).astype(np.uint8) * 255
                    test_mask = (test_mask > 127).astype(np.uint8) * 255
                    current_whole_mask = np.logical_or(current_whole_mask, test_mask).astype(np.uint8) * 255
                    count_tries = 11
                else:
                    count_tries += 1
                    
            if count_tries != 10:
                topLeft_bottomRight = [x_start * 512 / 1280, y_start * 512 / 1024, x_end * 512 / 1280, y_end * 512 / 1024]
                
                if Defect[chosen_defect_type.upper()].value == Defect.HOLE.value:
                    topLeft_bottomRight = guarantee_minimum_dimenions(*topLeft_bottomRight, min_dim_w=MIN_DIM_HOLE, min_dim_h=MIN_DIM_HOLE)
                
                data_faster_rcnn_item["boxes"].append(topLeft_bottomRight)

                data_faster_rcnn_item["labels"].append(Defect[chosen_defect_type.upper()].value)
                
                nodefect_image.paste(defect_image, (x_start - original_topleft_pos[1], y_start - original_topleft_pos[0]), defect_mask)
            
        data_faster_rcnn_item["boxes"] = torch.tensor(data_faster_rcnn_item["boxes"]).to(torch.float32)
        data_faster_rcnn_item["labels"] = torch.tensor(data_faster_rcnn_item["labels"]).to(torch.int64)
        
        data_faster_rcnn[synthetic_image_name] = data_faster_rcnn_item    
        
        # save
        nodefect_image.save(os.path.join(PATH_TO_SYNTHETIC, synthetic_image_name))
        Image.fromarray(current_whole_mask).convert('L').save(os.path.join(PATH_TO_SYNTHETIC_MASKS, f'{nodefect_name}_MDT_{i_th_sample}_mask.png'))   
        print(f'{i_th_sample + 1}/{num_tot_samples}', end='\r')
            
    pickle.dump(data_faster_rcnn, open(os.path.join(PATH_TO_SYNTHETIC, '..', 'data_faster_rcnn.pkl'), 'wb'))
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--tot_samples', type=int, help='Amount of synthetic images generated', dest='N_SAMPLES', default=500)
    parser.add_argument('--multi_defect', type=bool, help='Amount of synthetic images generated', dest='MULTI_DEFECT', default=False)    
    args = parser.parse_args()

    os.makedirs(PATH_TO_SYNTHETIC, exist_ok=True)
    os.makedirs(PATH_TO_SYNTHETIC_MASKS, exist_ok=True)
    os.makedirs(PATH_TO_SYNTHETIC_DEFECTS_HARMONIZED, exist_ok=True)
    delete_old_files()
    
    if args.MULTI_DEFECT:
        multi_defect_type_generation(args.N_SAMPLES)
    else:
        single_defect_type_generation(args.N_SAMPLES)
    