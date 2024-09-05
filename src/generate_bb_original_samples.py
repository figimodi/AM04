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
    data = {}

    for file_name in os.listdir(DEFECTS_DIR):
        key = f'{file_name}.jpg'
        data[key] = {
            'boxes': [],
            'labels': []
        }
        
        folders = [folder for folder in os.listdir(DEFECTS_DIR) if os.path.isdir(os.path.join(DEFECTS_DIR, folder)) and int(folder.split('Image')[-1]) < 50]
        
        for mask_folder in folders:
            for mask_file in os.listdir(os.path.join(DEFECTS_MASKS_DIR, mask_folder)):
                if '_PD_' in mask_file and str(file_name) == str(mask_file.split('_')[0]):
                    parts = mask_file.split('_')
                    label = Defect[parts[3].split('.')[0].upper()].value
                    
                    mask_path = os.path.join(DEFECTS_MASKS_DIR, mask_folder, mask_file)
                    mask = np.array(Image.open(mask_path).convert('L'))
                    true_points = np.nonzero(mask)
                    
                    try:  
                        min_y, min_x = np.min(true_points, axis=1) 
                        max_y, max_x = np.max(true_points, axis=1) 
                                            
                        border = [min_x * 512 / 1280, min_y * 512 / 1024, max_x * 512 / 1280, max_y * 512 / 1024]
                        
                        if label == 0: #HOLE
                            border = guarantee_minimum_dimenions(border, min_dim_w=MIN_DIM_HOLE, min_dim_h=MIN_DIM_HOLE)
                        elif label == 1: #VERTICAL
                            border = guarantee_minimum_dimenions(border, min_dim_w=MIN_W_VERTICAL)
                        elif label == 4: #HORIZONTAL
                            border = guarantee_minimum_dimenions(border, min_dim_h=MIN_H_HORIZONTAL)

                        data[key]['boxes'].append(border)
                        data[key]['labels'].append(label)
                    except:
                        print('Error in file:', os.path.join(DEFECTS_MASKS_DIR, mask_folder, mask_file))
        
        if len(data[key]['boxes']) > 0:
            data[key]['boxes'] = torch.Tensor(data[key]['boxes']).to(dtype=torch.float32)
            data[key]['labels'] = torch.Tensor(data[key]['labels']).to(dtype=torch.int64)
        else:
            del data[key]

    pickle.dump(data, open(os.path.join(DEFECTS_DIR, '..', 'data_original_faster_rcnn.pkl'), 'wb'))
    
if __name__ == '__main__':
    generate_pickle_faster_rcnn()