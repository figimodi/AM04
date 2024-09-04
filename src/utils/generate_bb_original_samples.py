from enum import Enum
import os
from PIL import Image
import pickle
import numpy as np
import torch

DEFECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Defects')
DEFECTS_MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'DefectsMasks')

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
            'borders': [],
            'labels': []
        }
        
        folders = [folder for folder in os.listdir(DEFECTS_DIR) if os.path.isdir(os.path.join(DEFECTS_DIR, folder))]
        
        for mask_folder in folders:
            for mask_file in os.listdir(os.path.join(DEFECTS_MASKS_DIR, mask_folder)):
                if '_PD_' in mask_file and file_name in mask_file:
                    parts = mask_file.split('_')
                    label = Defect[parts[3].split('.')[0].upper()].value
                    
                    mask_path = os.path.join(DEFECTS_MASKS_DIR, mask_folder, mask_file)
                    mask = np.array(Image.open(mask_path).convert('L'))
                    true_points = np.argwhere(mask==255)
                    
                    try:  
                        top_left = true_points.min(axis=0)
                        bottom_right = true_points.max(axis=0)
                        
                        border = [top_left[0] * 512 / 1280, top_left[1] * 512 / 1024, bottom_right[0] * 512 / 1280, bottom_right[1] * 512 / 1024]
                        
                        data[key]['borders'].append(border)
                        data[key]['labels'].append(label)
                    except:
                        print('Error in file:', os.path.join(DEFECTS_MASKS_DIR, mask_folder, mask_file))
        
        if len(data[key]['borders']) > 0:
            data[key]['borders'] = torch.Tensor(data[key]['borders']).to(dtype=torch.float32)
            data[key]['labels'] = torch.Tensor(data[key]['labels']).to(dtype=torch.int64)
        else:
            del data[key]

    pickle.dump(data, open(os.path.join(DEFECTS_DIR, '..', 'data_original_faster_rcnn.pkl'), 'wb'))
    
if __name__ == '__main__':
    generate_pickle_faster_rcnn()