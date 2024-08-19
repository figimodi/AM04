import os
from PIL import Image
import numpy as np
CUR_DIR = [...] ## set directory here

for image_name in os.listdir(CUR_DIR):
    if os.path.isdir(os.path.join(CUR_DIR, image_name)):
        for image in os.listdir(os.path.join(CUR_DIR, image_name)):
            cur_image = Image.open(os.path.join(CUR_DIR, image_name, image)).convert("L")
            cur_image = Image.fromarray((np.array(cur_image)> 127).astype(np.uint8) * 255)
            os.remove(os.path.join(CUR_DIR, image_name, image))
            new_path = os.path.join(CUR_DIR, image_name, image).replace('.jpg', '.png')
            cur_image.convert("L").save(new_path)
            