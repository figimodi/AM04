import os
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
                if os.path.isfile(file_path) and ( '_ct_' in filename):
                    os.remove(file_path)

    for defect_folder in os.listdir(PATH_MASK):
        if os.path.isdir(os.path.join(PATH_MASK, defect_folder)):
            for comb_mask_file in os.listdir(PATH_MASK):
                if '_CB_' in comb_mask_file:
                    mask_comb_path = os.path.join(PATH_MASK, defect_folder, comb_mask_file)
                    os.remove(mask_comb_path)           
                    
    print('Old color transferred images deleted!')
    
def main():
    #No laser, no whole mask
    masks_filenames = [ os.path.join(mask_folder, mask_filename) for mask_folder in os.listdir(PATH_MASK) if os.path.isdir(os.path.join(PATH_MASK, mask_folder)) and mask_filename.endswith('.jpg') and '_L_' not in mask_filename and '_mask' not in mask_filename for mask_filename in os.listdir(os.path.join(PATH_MASK, mask_folder))]
    defect_folders = [ defect_folder for defect_folder in os.listdir(PATH_DEFECTS) if os.path.isdir(os.path.join(PATH_DEFECTS, defect_folder))]
    
    print('Generating color transferred images...')
    print(f'0/{len(defect_folders)}', end='\r')
    
    #iterate image defects
    for i_df, defect_folder in enumerate(defect_folders):
        # mask paths to current image defect
        current_defect_masks = [mask_filename for mask_filename in masks_filenames if mask_filename.startswith(f'{defect_folder}_') ]
        all_current_defect_masks = []
        
        #generate all possible combinations of mask defects for current defect image
        for i_cdm in range(1, len(current_defect_masks)+1):
            mask_combinations_i_cdm = combinations(current_defect_masks, i_cdm)    
            all_current_defect_masks.extend(mask_combinations_i_cdm) # list of sublists (each sublist tells the combination of defect masks)
        
        maximum_pd = max([ int(mask_filename.split('_')[2]) for mask_filename in os.listdir(os.path.join(PATH_MASK, defect_folder)) if '_PD_' in mask_filename])
        
        # for each mask apply the resulting mask and color transfer
        for i_acdm, mask_filename_comb in enumerate(all_current_defect_masks):
            
            current_pd = -1
            
            if len(mask_filename_comb) == 1: 
                mask_path = os.path.join(PATH_MASK, mask_filename_comb[0])
                current_pd = int(mask_filename_comb[0].split('_')[2])
                mask = Image.open(mask_path).convert("L")                     
            else:
                mask_paths = [os.path.join(PATH_MASK, mask_filename_comb[i_mfc]) for i_mfc in range(len(mask_filename_comb))] 
                masks = [np.array(Image.open(mask_paths[i_mp]).convert("L")) for i_mp in range(len(mask_paths))]
                final_mask = masks[0]
                for i_m in range(1, len(masks)):
                    final_mask = np.where(masks[i_m] > 0, 255, final_mask)
                mask = Image.fromarray(final_mask).convert("L")
                
                mask.save(os.path.join(PATH_MASK, defect_folder, f'{defect_folder}_CB_{i_acdm}.jpg'))

            defect_path = os.path.join(PATH_DEFECTS, defect_folder, f'{defect_folder}.jpg')
            
            defect = Image.open(defect_path).convert('RGB')
            
            manipulated_defects = apply_color_transfer(defect, mask)
            
            for i_md, md in enumerate(manipulated_defects):
                md.save(os.path.join(PATH_DEFECTS, defect_folder ,f'{defect_folder}_CT_{current_pd if current_pd > 0 else i_acdm + maximum_pd + 1}_{i_md}.jpg'))
                
        print(f'{i_df+1}/{len(defect_folders)}', end='\r')
        
    print('\nDone!')
    

if __name__ == "__main__":
    os.makedirs(PATH_MASK_COMBINATIONS, exist_ok=True)
    delete_old_files()
    #main()