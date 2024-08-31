import os
from PIL import Image
import pickle
import numpy as np

DEFECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Defects')
DEFECTS_MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'DefectsMasks')

def generate_pickle_faster_rcnn():
    data = {}

    for file_name in os.listdir(DEFECTS_DIR):
        
        data[file_name] = {
            'borders': [],
            'labels': []
        }
        
        for mask_folder in os.listdir(DEFECTS_MASKS_DIR):
            for mask_file in os.listdir(os.path.join(DEFECTS_MASKS_DIR, mask_folder)):
                if '_PD_' in mask_file and file_name in mask_file:
                    parts = mask_file.split('_')
                    label = parts[2].split('.')[0]
                    
                    mask_path = os.path.join(DEFECTS_MASKS_DIR, mask_folder, mask_file)
                    mask = np.array(Image.open(mask_path).convert('L'))
                    true_points = np.argwhere(mask==255)
                    
                    try:  
                        top_left = true_points.min(axis=0)
                        bottom_right = true_points.max(axis=0)
                        
                        border = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
                        
                        data[file_name]['borders'].append(border)
                        data[file_name]['labels'].append(label)
                    except:
                        print('Error in file:', os.path.join(DEFECTS_MASKS_DIR, mask_folder, mask_file))

    pickle.dump(data, open(os.path.join(DEFECTS_DIR, 'data_original_faster_rcnn.pkl'), 'wb'))
    
if __name__ == '__main__':
    generate_pickle_faster_rcnn()