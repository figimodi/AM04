import shutil
import os

PATH_TO_MASKS = os.path.join(os.getcwd(), 'data', 'DefectsMasks')
PATH_TO_DEFECTS = os.path.join(os.getcwd(), 'data', 'Defects')
masks_filenames = [mask_filename for mask_filename in os.listdir(PATH_TO_MASKS) if mask_filename.endswith('.jpg')]
defects_filenames_noext = [defect_filename.split('.')[0] for defect_filename in os.listdir(PATH_TO_DEFECTS) if not defect_filename.startswith('Image0') ]

print(masks_filenames)
print(defects_filenames_noext)

for df in defects_filenames_noext:
    for mf in masks_filenames:
        mf_parts = mf.split('_')
        shutil.copyfile( os.path.join( PATH_TO_MASKS, mf ), os.path.join( PATH_TO_MASKS, f'{df}_{mf_parts[1]}_{mf_parts[2]}' ) )
