from enum import Enum
import os
from PIL import Image
import pickle
import numpy as np
import torch
from utils.guarantee_minimum_dimensions import guarantee_minimum_dimenions

DEFECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Defects')
DEFECTS_MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'DefectsMasks')

MIN_DIM_HOLE = 32
MIN_H_HORIZONTAL = 15
MIN_W_VERTICAL = 15

class Defect(Enum):
    HOLE = 0
    VERTICAL = 1
    INCANDESCENCE = 2
    SPATTERING = 3
    HORIZONTAL = 4

def generate_pickle_faster_rcnn():
    data_pkl = {}
    defect_masks_folders_list = [folder for folder in os.listdir(DEFECTS_DIR) if os.path.isdir(os.path.join(DEFECTS_DIR, folder)) and int(folder.split('Image')[-1]) < 50]
    
    print("Generating pickle file...")

    #For each defect image
    for defect_mask_folder in defect_masks_folders_list:
        key = f'{defect_mask_folder}.jpg'
        mask_filenames_list = [mask_filename for mask_filename in os.listdir(os.path.join(DEFECTS_MASKS_DIR, defect_mask_folder)) if '_PD_' in mask_filename and str(defect_mask_folder) == str(mask_filename.split('_')[0])]
        
        for mask_filename in mask_filenames_list:
            if data_pkl.get(key) is None:
                data_pkl[key] = {'boxes': [], 'labels': []}
            
            label = Defect[mask_filename.split('_')[3].split('.')[0].upper()].value
            
            mask_path = os.path.join(DEFECTS_MASKS_DIR, defect_mask_folder, mask_filename)
            mask = np.array(Image.open(mask_path).convert('L'))
            
            foreground = np.nonzero(mask)
            
            min_y, min_x = np.min(foreground, axis=1) 
            max_y, max_x = np.max(foreground, axis=1) 
                                
            topLeft_bottomRight = [min_x * 512 / 1280, min_y * 512 / 1024, max_x * 512 / 1280, max_y * 512 / 1024]
            
            if label == 0: #HOLE
                topLeft_bottomRight = guarantee_minimum_dimenions(*topLeft_bottomRight, min_dim_w=MIN_DIM_HOLE, min_dim_h=MIN_DIM_HOLE)
            elif label == 1: #VERTICAL
                topLeft_bottomRight = guarantee_minimum_dimenions(*topLeft_bottomRight, min_dim_w=MIN_W_VERTICAL)
            elif label == 4: #HORIZONTAL
                topLeft_bottomRight = guarantee_minimum_dimenions(*topLeft_bottomRight, min_dim_h=MIN_H_HORIZONTAL)

            data_pkl[key]['boxes'].append(topLeft_bottomRight)
            data_pkl[key]['labels'].append(label)
        
        data_pkl[key]['boxes'] = torch.Tensor(data_pkl[key]['boxes']).to(dtype=torch.float32)
        data_pkl[key]['labels'] = torch.Tensor(data_pkl[key]['labels']).to(dtype=torch.int64)

    pickle.dump(data_pkl, open(os.path.join(DEFECTS_DIR, '..', 'data_original_faster_rcnn.pkl'), 'wb'))
    
    print("Done!")
    
if __name__ == '__main__':
    generate_pickle_faster_rcnn()