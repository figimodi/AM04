import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PATH_TO_SYNTHETIC = os.path.join(ROOT_DIR, 'data', 'SyntheticDefects')
PATH_TO_SYNTHETIC_MASKS = os.path.join(ROOT_DIR, 'data', 'SyntheticDefectsMasks')
PATH_MASK = os.path.join(ROOT_DIR, 'data', 'DefectsMasks')
PATH_DEFECTS = os.path.join(ROOT_DIR, 'data', 'Defects')

def main():
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
    
    for fn in os.listdir(PATH_TO_SYNTHETIC):
        file_path = os.path.join(PATH_TO_SYNTHETIC, fn)
        os.remove(file_path)   
        
    for fn in os.listdir(PATH_TO_SYNTHETIC_MASKS):
        file_path = os.path.join(PATH_TO_SYNTHETIC_MASKS, fn)
        os.remove(file_path)        
                    
    print('Old synthetic images deleted!')

if __name__ == "__main__":
    main()