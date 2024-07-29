import os
import random
from PIL import Image
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PATH_TO_MASKS = os.path.join(ROOT_DIR, 'data', 'DefectsMasks')
PATH_TO_DEFECTS = os.path.join(ROOT_DIR, 'data', 'Defects')
PATH_TO_NODEFECTS = os.path.join(ROOT_DIR, 'data', 'NoDefects')
PATH_TO_SYNTHETIC = os.path.join(ROOT_DIR, 'data', 'Synthetic')

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

def main():
    
    masks_filenames = [f for f in os.listdir(PATH_TO_MASKS) if f.endswith('.jpg')]
    nodefects_filenames = [f for f in os.listdir(PATH_TO_NODEFECTS) if f.endswith('.jpg')]
    defect_types = list(set([f.split('_')[2].split('.')[0] for f in masks_filenames]))
    
    ## paramaters
    
    # how many sinthetic images with defect generated per NoDefect image
    augmentation_factor = 5
    # probability of having few defects against having many defects (few and many defined next)
    probability_few_defects= .8
    few = lambda: random.choice([1,2,3])
    many = lambda: random.choice([4,5,6])
    # probability_rotate = .5
    # probability_flip = .5
    choose_how_many_defects = lambda: few() if random.random() < probability_few_defects else many()
    choose_defect_type = lambda: random.choice(defect_types)
    get_defect_mask_filename_by_type = lambda chosen_type: random.choice([f for f in masks_filenames if chosen_type in f])
    
    print('Generating synthetic images...')
    
    print(f'0/{len(nodefects_filenames)}', end='\r')
    for nodef_i, fn_nd in enumerate(nodefects_filenames):
        path_to_nodefect_image = os.path.join(PATH_TO_NODEFECTS, fn_nd)
        nodefect_image = Image.open(path_to_nodefect_image).convert('RGB')
        nodefect_name = fn_nd.split('.')[0]
        for aug_i in range(augmentation_factor):
            nodefect_image_current = nodefect_image.copy()
            number_of_defects = choose_how_many_defects()
            current_mask = None
            for def_i in range(number_of_defects):
                if def_i == 0:
                    #current mask store a union of all masks of the chosen defects, It will be used to check if any defect overlaps
                    current_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                
                #defect type that will be inserted in the sinthetic image
                chosen_defect_type = choose_defect_type()
                
                #choose which defect will be used (so choose the file of its mask)
                defect_mask_filename = get_defect_mask_filename_by_type(chosen_defect_type)
                path_to_defect_mask = os.path.join(PATH_TO_MASKS, defect_mask_filename)
                #get in which image is this defect placed
                defect_name = defect_mask_filename.split('_')[0]
                
                #read the defect image
                path_to_defect_image = os.path.join(PATH_TO_DEFECTS, defect_name, f'{defect_name}.jpg')
                defect_image = Image.open(path_to_defect_image).convert('RGB')
                
                # read the defect mask
                defect_mask = Image.open(path_to_defect_mask).convert('L')
                #get the top left coordinates of the mask
                original_topleft_pos = get_pos_topleft(np.array(defect_mask))
                #the a numpy copy of the mask (used for checking any overlap)
                proto_cropped_mask = crop_image(np.array(defect_mask))
                
                keep_trying = True
                
                while keep_trying == True:
                    #choose a differebt top left position (translation)
                    x_start = random.randint(max(FRAME_BORDERS['TL'][0], FRAME_BORDERS['BL'][0]), min(FRAME_BORDERS['TR'][0], FRAME_BORDERS['BR'][0])-proto_cropped_mask.shape[1])
                    y_start = random.randint(max(FRAME_BORDERS['TL'][1], FRAME_BORDERS['TR'][1]), min(FRAME_BORDERS['BL'][1], FRAME_BORDERS['BR'][1])-proto_cropped_mask.shape[0])
                    
                    #we will use this in order to move the defect to the chosen position e check then if there are any overlapping with the defects chosen so far
                    test_mask = np.zeros((nodefect_image.size[1], nodefect_image.size[0]), dtype=np.uint8)
                    
                    x_end = x_start + proto_cropped_mask.shape[1]
                    y_end = y_start + proto_cropped_mask.shape[0]
                    
                    test_mask[y_start:y_end, x_start:x_end] = proto_cropped_mask
                    
                    if not np.any(test_mask*current_mask) or def_i==0:
                        keep_trying = False
                        current_mask = np.logical_or(current_mask, test_mask)*255
                
                #past the cropped defect in the original image
                nodefect_image_current.paste(defect_image, (x_start - original_topleft_pos[1], y_start - original_topleft_pos[0]), defect_mask)
                
            # save
            nodefect_image_current.save(os.path.join(PATH_TO_SYNTHETIC, f'{nodefect_name}_{aug_i}.jpg'))   
            print(f'{nodef_i+1}/{len(nodefects_filenames)}', end='\r')
            
    print('Done!')

if __name__ == '__main__':
    os.makedirs(PATH_TO_SYNTHETIC,exist_ok=True)
    delete_old_files()
    # main()