from enum import Enum
import os
from PIL import Image
import pickle
import numpy as np
import torch

DEFECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Defects')
DEFECTS_MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'DefectsMasks')

MIN_DIM_HOLE = 40
MIN_H_HORIZONTAL = 30
MIN_W_VERTICAL = 10

class Defect(Enum):
    HOLE = 0
    VERTICAL = 1
    INCANDESCENCE = 2
    SPATTERING = 3
    HORIZONTAL = 4

def guarantee_minimum_dimenions(min_x, min_y, max_x, max_y, min_dim_w=None, min_dim_h=None):
    if min_dim_w:
        width = max_x - min_x
        
        inc_x = ((min_dim_w - width) // 2) if (min_dim_w - width // 2) > 0 else 0

        min_x -= inc_x
        max_x += inc_x
        
        max_x = max_x + 1 if (min_dim_w - width)%2 == 1 else max_x
        
    if min_dim_h:        
        height = max_y - min_y
        
        inc_y = ((min_dim_h - height) // 2) if (min_dim_h - height // 2) > 0 else 0
        
        min_y -= inc_y
        max_y += inc_y
        
        max_y = max_y + 1 if (min_dim_h - height)%2 == 1 else max_y

    return min_x, min_y, max_x, max_y
    

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
                        
                        if label == 0: #HOLE
                            min_x, min_y, max_x, max_y = guarantee_minimum_dimenions(min_x, min_y, max_x, max_y, min_dim_w=MIN_DIM_HOLE, min_dim_h=MIN_DIM_HOLE)
                        elif label == 1: #VERTICAL
                            min_x, min_y, max_x, max_y = guarantee_minimum_dimenions(min_x, min_y, max_x, max_y, min_dim_w=MIN_W_VERTICAL)
                        elif label == 4: #HORIZONTAL
                            min_x, min_y, max_x, max_y = guarantee_minimum_dimenions(min_x, min_y, max_x, max_y, min_dim_h=MIN_H_HORIZONTAL)
                              
                        border = [min_x * 512 / 1280, min_y * 512 / 1024, max_x * 512 / 1280, max_y * 512 / 1024]
                        
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