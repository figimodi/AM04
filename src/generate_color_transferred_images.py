import os
import random
from utils.color_transfering import apply_color_transfer
from itertools import combinations
from PIL import Image
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PATH_MASK = os.path.join(ROOT_DIR, 'data', 'DefectsMasks')
PATH_DEFECTS = os.path.join(ROOT_DIR, 'data', 'Defects')

def delete_old_files():    
    for dir_image in os.listdir(PATH_DEFECTS):
        if os.path.isdir(os.path.join(PATH_DEFECTS, dir_image)):
            for filename in os.listdir(os.path.join(PATH_DEFECTS, dir_image)):
                file_path = os.path.join(PATH_DEFECTS, dir_image, filename)
                if os.path.isfile(file_path) and ( '_CT_' in filename):
                    os.remove(file_path)

    for defect_folder in os.listdir(PATH_MASK):
        if os.path.isdir(os.path.join(PATH_MASK, defect_folder)):
            for comb_mask_file in os.listdir(os.path.join(PATH_MASK, defect_folder)):
                if '_CB_' in comb_mask_file:
                    mask_comb_path = os.path.join(PATH_MASK, defect_folder, comb_mask_file)
                    os.remove(mask_comb_path)           
                    
    print('Old color transferred images deleted!')
    
def main():
    mask_folders_list = [ mask_folder for mask_folder in os.listdir(PATH_DEFECTS) if os.path.isdir(os.path.join(PATH_DEFECTS, mask_folder))]
    
    print('Generating color transferred images...')
    print(f'0/{len(mask_folders_list)}', end='\r')
    
    for i_mf, mask_folder in enumerate(mask_folders_list):
        folder_number = int(mask_folder.split('Image')[-1])
        masks_filenames_list = [ mask_filename for mask_filename in os.listdir(os.path.join(PATH_MASK, mask_folder)) if mask_filename.endswith('.png') and '_PD_' in mask_filename and not ( folder_number < 50 and 'Spattering' in mask_filename ) ]
        
        if len(masks_filenames_list) == 0:
            continue
        
        masks_filenames_combinations_list = []
            
        if len(masks_filenames_list) > 4:
            masks_filenames_combinations_list.extend(combinations(masks_filenames_list, 1))
        else:
            for i_cdm in range(1, 2 if len(masks_filenames_list) == 1 else len(masks_filenames_list)):
                mask_combinations_i_cdm = combinations(masks_filenames_list, i_cdm)    
                masks_filenames_combinations_list.extend(mask_combinations_i_cdm) # list of sublists (each sublist tells the combination of defect masks)
                
        if len(masks_filenames_list) > 1 and os.path.isfile(os.path.join(PATH_MASK, mask_folder, f'{mask_folder}_Mask_00.png')):
            masks_filenames_combinations_list.append([ f'{mask_folder}_Mask_00.png' ] )
        
        highest_mask_id = max([int(mask_filename.split('_')[2].split('.')[0]) for mask_filename in masks_filenames_list])
        combination_counter = 0
        
        # for each mask apply the resulting mask and color transfer
        for i_acdm, masks_filenames_combinations in enumerate(masks_filenames_combinations_list):
            
            mask_id = None
            
            if len(masks_filenames_combinations) == 1: 
                mask_path = os.path.join(PATH_MASK, mask_folder, masks_filenames_combinations[0])
                mask_id = int(masks_filenames_combinations[0].split('_')[2].split('.')[0])
                mask = Image.open(mask_path).convert("L")                     
            else:
                masks = [ np.array( Image.open( os.path.join( PATH_MASK, mask_folder, mask_filename ) ).convert("L")) for mask_filename in masks_filenames_combinations ] 
                
                final_mask = masks[0]
                
                for i_m in range(1, len(masks)):
                    final_mask = (final_mask > 127).astype(np.uint8) * 255
                    masks[i_m] = (masks[i_m] > 127).astype(np.uint8) * 255
                    final_mask = np.logical_or(final_mask, masks[i_m]).astype(np.uint8) * 255

                final_mask = Image.fromarray(final_mask).convert("L")
                
                mask_id = highest_mask_id+combination_counter+1
                
                combination_counter += 1
                
                final_mask.save(os.path.join(PATH_MASK, mask_folder, f'{mask_folder}_CB_{mask_id}.png'))

            defect = Image.open(os.path.join(PATH_DEFECTS, mask_folder, f'{mask_folder}.jpg')).convert('RGB')
            
            manipulated_defects = apply_color_transfer(defect, mask)
            
            for i_md, md in enumerate(manipulated_defects):
                md.save(os.path.join(PATH_DEFECTS, mask_folder ,f'{mask_folder}_CT_{mask_id}_{i_md}.jpg'))
                
        print(f'{i_mf+1}/{len(mask_folders_list)}', end='\r')
        
    print('\nDone!')
    

if __name__ == "__main__":
    delete_old_files()
    main()
    