import os

from utils.color_transfering import apply_color_transfer

PATH_MASK = os.path.join('data', 'DefectsMasks')
PATH_DEFECTS = os.path.join('data', 'Defects')

def delete_color_transferred_images():
    for dir_image in os.listdir(PATH_DEFECTS):
        if os.path.isdir(os.path.join(PATH_DEFECTS, dir_image)):
            for filename in os.listdir(os.path.join(PATH_DEFECTS, dir_image)):
                file_path = os.path.join(PATH_DEFECTS, dir_image, filename)
                if os.path.isfile(file_path) and '_ct_' in filename:
                    os.remove(file_path)
                    
    print('Old color transferred images deleted!')
def main():
    masks_filenames = [filename for filename in os.listdir(PATH_MASK) if filename.endswith('.jpg')]
    
    print('Generating color transferred images...')
    print(f'0/{len(masks_filenames)}', end='\r')
    for i, mf in enumerate(masks_filenames):
        
        image_name = mf.split('_')[0]
        
        mask_path = os.path.join(PATH_MASK, mf)
        defect_path = os.path.join(PATH_DEFECTS, image_name, f'{image_name}.jpg')
        
        color_transferred_images = apply_color_transfer(defect_path, mask_path)
        
        for j, cti in enumerate(color_transferred_images):
            cti.save(os.path.join(PATH_DEFECTS, image_name ,f'{image_name}_ct_{j}.jpg'))
            
        print(f'{i+1}/{len(masks_filenames)}', end='\r')
        
    print('\nDone!')
    

if __name__ == "__main__":
    delete_color_transferred_images()
    main()