import os


map = { 
    'Incandesccence': 'Incandescence',
    'Incandescence': 'Incandescence', 
    'Horizontal': 'Horizontal', 
    'Spatttering': 'Spattering',
    'Spattering': 'Spattering', 
    'Layer': 'Layer',
    'incandescence': 'Incandescence', 
    'Hole': 'Hole', 
    'Vertical': 'Vertical'
}

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PATH_TO_MASKS = os.path.join(ROOT_DIR, 'data', 'DefectsMasks')

masks_paths = [ os.path.join(PATH_TO_MASKS, mask_folder, mask_filename) for mask_folder in os.listdir(PATH_TO_MASKS) if os.path.isdir(os.path.join(PATH_TO_MASKS, mask_folder)) for mask_filename in os.listdir(os.path.join(PATH_TO_MASKS, mask_folder)) if mask_filename.endswith('.jpg') and '_L_' not in mask_filename and '_Mask_' not in mask_filename]

for p in masks_paths:
    if os.path.exists():
        parts = p.split('_')
        parts[-1] = f'{map[parts[-1].split(".")[0]]}.jpg'
        os.rename(p, '_'.join(parts))