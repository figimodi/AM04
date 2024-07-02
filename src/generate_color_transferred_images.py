import os
from utils.color_transfering import apply_color_transfer
from itertools import combinations
from PIL import Image
import numpy as np

PATH_MASK = os.path.join(os.getcwd(), 'data', 'DefectsMasks')
PATH_DEFECTS = os.path.join(os.getcwd(), 'data', 'Defects')
PATH_MASK_COMBINATIONS = os.path.join(PATH_MASK, 'combinations')

def delete_old_files():
    for dir_image in os.listdir(PATH_DEFECTS):
        if os.path.isdir(os.path.join(PATH_DEFECTS, dir_image)):
            for filename in os.listdir(os.path.join(PATH_DEFECTS, dir_image)):
                file_path = os.path.join(PATH_DEFECTS, dir_image, filename)
                if os.path.isfile(file_path) and ( '_ct_' in filename or 'prova' in filename ):
                    os.remove(file_path)
                    
    for comb_mask in os.listdir(PATH_MASK_COMBINATIONS):
        file_path = os.path.join(PATH_MASK_COMBINATIONS, comb_mask)
        os.remove(file_path)           
                    
    print('Old color transferred images deleted!')
    
def main():
    masks_filenames = [ mask_filename for mask_filename in os.listdir(PATH_MASK) if mask_filename.endswith('.jpg') ]
    defect_filenames_noext = [ defect_filename.split('.')[0] for defect_filename in os.listdir(PATH_DEFECTS) ]
    
    print('Generating color transferred images...')
    print(f'0/{len(defect_filenames_noext)}', end='\r')
    
    #iterate defects
    for i, defect_filename in enumerate(defect_filenames_noext):
        current_defect_masks = [mask_filename for mask_filename in masks_filenames if mask_filename.startswith(f'{defect_filename}_') ]
        all_current_defect_masks = []
        
        #generate all possible cominations of masks
        for k in range(1, len(current_defect_masks)+1):
            mask_combinations_k = combinations(current_defect_masks, k)    
            all_current_defect_masks.extend(mask_combinations_k)
        
        # for each mask combination apply the resulting mask and color transfer
        for i_acdm, mask_filename_comb in enumerate(all_current_defect_masks):
            
            if len(mask_filename_comb) == 1:
                mask_path = os.path.join(PATH_MASK, mask_filename_comb[0])
                mask = Image.open(mask_path).convert("L")                     
            else:
                mask_paths = [os.path.join(PATH_MASK, mask_filename_comb[i_mfc]) for i_mfc in range(len(mask_filename_comb))] 
                masks = [np.array(Image.open(mask_paths[i_mp]).convert("L")) for i_mp in range(len(mask_paths))]
                final_mask = masks[0]
                for i_m in range(1, len(masks)):
                    final_mask = np.where(masks[i_m] > 0, 255, final_mask)
                mask = Image.fromarray(final_mask).convert("L")
                
                mask_path = os.path.join(PATH_MASK, mask_filename_comb[0])
                mask.save(os.path.join(PATH_MASK, 'combinations', f'{defect_filename}_comb_{i_acdm}.jpg'))

            defect_path = os.path.join(PATH_DEFECTS, defect_filename, f'{defect_filename}.jpg')
            
            defect = Image.open(defect_path).convert('RGB')
            
            manipulated_defects = apply_color_transfer(defect, mask)
            
            for i_md, md in enumerate(manipulated_defects):
                md.save(os.path.join(PATH_DEFECTS, defect_filename ,f'{defect_filename}_ct_{i_acdm}_{i_md}.jpg'))
                
        print(f'{i+1}/{len(defect_filenames_noext)}', end='\r')
        
    print('\nDone!')
    

if __name__ == "__main__":
    delete_old_files()
    main()