import os
import random
from PIL import Image
import numpy as np
import argparse

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PATH_TO_MASKS = os.path.join(ROOT_DIR, 'data', 'DefectsMasks')
PATH_TO_DEFECTS = os.path.join(ROOT_DIR, 'data', 'Defects')
PATH_TO_NODEFECTS = os.path.join(ROOT_DIR, 'data', 'NoDefects')
PATH_TO_SYNTHETIC = os.path.join(ROOT_DIR, 'data', 'SyntheticDefects')
PATH_TO_SYNTHETIC_MASKS = os.path.join(ROOT_DIR, 'data', 'SyntheticDefectsMasks')
PATH_TO_SYNTHETIC_DEFECTS_HARMONIZED = os.path.join(ROOT_DIR, 'data', 'SyntheticDefectsHarmonized')

#borders of the frame. X and Y coordinates (origin is TOP LEFT). Taken using paint (while moving the cursor gives x and y coordinates)
FRAME_BORDERS = {
    'TL': (240, 120),
    'TR': (1275, 130),
    'BL': (170, 885),
    'BR': (1275, 880)
}

def delete_old_files():
    for fn in os.listdir(PATH_TO_SYNTHETIC):
        file_path = os.path.join(PATH_TO_SYNTHETIC, fn)
        os.remove(file_path)   
        
    for fn in os.listdir(PATH_TO_SYNTHETIC_MASKS):
        file_path = os.path.join(PATH_TO_SYNTHETIC_MASKS, fn)
        os.remove(file_path)        
                    
    print('Old color transferred images deleted!')

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

def main(samples_number_per_defect = 10, probability_few_defects = .8):
    # probability of having few defects against having many defects (few and many defined next)
    masks_paths = [ os.path.join(PATH_TO_MASKS, mask_folder, mask_filename) for mask_folder in os.listdir(PATH_TO_MASKS) if os.path.isdir(os.path.join(PATH_TO_MASKS, mask_folder)) for mask_filename in os.listdir(os.path.join(PATH_TO_MASKS, mask_folder)) if (mask_filename.endswith('.jpg') or mask_filename.endswith('.png')) and '_PD_' in mask_filename and not(int(mask_folder.split('Image')[-1]) < 50 and 'Spattering' in mask_filename ) ]
    nodefects_filenames = [f for f in os.listdir(PATH_TO_NODEFECTS) if f.endswith('.jpg') or f.endswith('.png')]
    defect_types = list(set([f.split('_')[3].split('.')[0] for f in masks_paths]))

    few = lambda: random.choice([1,2,3])
    many = lambda: random.choice([4,5,6])

    choose_how_many_defects = lambda: few() if random.random() < probability_few_defects else many()
    # choose_defect_type = lambda: random.choice(defect_types)
    get_defect_mask_path_by_type = lambda chosen_type: random.choice([p for p in masks_paths if (p.endswith('.jpg') or p.endswith('.png')) and p.split('_')[-1].split('.')[0] == chosen_type])
    
    print('Generating synthetic images...')
    
    print(f'0/{samples_number_per_defect*len(defect_types)}', end='\r')
    
    for i_dt, chosen_defect_type in enumerate(defect_types):
        for i_sndp in range(samples_number_per_defect):
            nodefects_filename = random.choice(nodefects_filenames)
            path_to_nodefect_image = os.path.join(PATH_TO_NODEFECTS, nodefects_filename)
            nodefect_image = Image.open(path_to_nodefect_image).convert('RGB')
            nodefect_name = nodefects_filename.split('.')[0]
            
            nodefect_image_current = nodefect_image.copy()
            number_of_defects = choose_how_many_defects()
            current_mask = None
            
            for def_i in range(number_of_defects):
                if def_i == 0:
                    #current mask store a union of all masks of the chosen defects, It will be used to check if any defect overlaps
                    current_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                
                #choose which defect will be used (so choose the file of its mask)
                defect_mask_path = get_defect_mask_path_by_type(chosen_defect_type)
                #get in which image is this defect placed
                defect_name =  os.path.basename(defect_mask_path).split('_')[0]
                
                #read the defect image
                path_to_defect_image = os.path.join(PATH_TO_DEFECTS, defect_name, f'{defect_name}.jpg')
                defect_image = Image.open(path_to_defect_image).convert('RGB')
                
                # read the defect mask
                defect_mask = Image.open(defect_mask_path).convert('L')
                #get the top left coordinates of the mask
                original_topleft_pos = get_pos_topleft(np.array(defect_mask))
                #the a numpy copy of the mask (used for checking any overlap)
                proto_cropped_mask = crop_image(np.array(defect_mask))
                
                keep_trying = True
                count_tryies = 0
                
                while keep_trying == True and count_tryies < 10:
                    #choose a differebt top left position (translation)
                    try:
                        x_start = random.randint(max(FRAME_BORDERS['TL'][0], FRAME_BORDERS['BL'][0]), min(FRAME_BORDERS['TR'][0], FRAME_BORDERS['BR'][0])-proto_cropped_mask.shape[1])
                        y_start = random.randint(max(FRAME_BORDERS['TL'][1], FRAME_BORDERS['TR'][1]), min(FRAME_BORDERS['BL'][1], FRAME_BORDERS['BR'][1])-proto_cropped_mask.shape[0])
                    except:
                        count_tryies+=1
                        continue
                    
                    #we will use this in order to move the defect to the chosen position e check then if there are any overlapping with the defects chosen so far
                    test_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                    
                    x_end = x_start + proto_cropped_mask.shape[1]
                    y_end = y_start + proto_cropped_mask.shape[0]
                    
                    test_mask[y_start:y_end, x_start:x_end] = proto_cropped_mask
                    
                    if not np.any(test_mask*current_mask) or def_i==0:
                        keep_trying = False
                        current_mask = (current_mask > 127).astype(np.uint8) * 255
                        test_mask = (test_mask > 127).astype(np.uint8) * 255
                        current_mask = np.logical_or(current_mask, test_mask).astype(np.uint8) * 255
                        
                    count_tryies+=1
                
                #past the cropped defect in the original image
                nodefect_image_current.paste(defect_image, (x_start - original_topleft_pos[1], y_start - original_topleft_pos[0]), defect_mask)
                
            # save
            nodefect_image_current.save(os.path.join(PATH_TO_SYNTHETIC, f'{nodefect_name}_{chosen_defect_type}_{i_sndp}.jpg'))
            Image.fromarray(current_mask).convert('L').save(os.path.join(PATH_TO_SYNTHETIC_MASKS, f'{nodefect_name}_{chosen_defect_type}_{i_sndp}_mask.png'))   
            print(f'{i_sndp + i_dt*samples_number_per_defect + 1}/{samples_number_per_defect*len(defect_types)}', end='\r')
            
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--tot_samples', type=int, help='Amount of synthetic images generated', required=True, dest='N_SAMPLES')    
    args = parser.parse_args()

    os.makedirs(PATH_TO_SYNTHETIC, exist_ok=True)
    os.makedirs(PATH_TO_SYNTHETIC_MASKS, exist_ok=True)
    os.makedirs(PATH_TO_SYNTHETIC_DEFECTS_HARMONIZED, exist_ok=True)
    delete_old_files()
    main(samples_number_per_defect=args.N_SAMPLES/5)