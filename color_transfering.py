import cv2
import numpy as np
import random
from PIL import Image,ImageEnhance
import os

def apply_color_transfer(image, mask):
    masked_image = Image.new("RGB", image.size, (0, 0, 0))
    masked_image.paste(image, (0, 0), mask)  # Paste modifies in place
    
    br_enhancer = ImageEnhance.Brightness(masked_image)
    co_enhancer = ImageEnhance.Contrast(masked_image)
    sa_enhancer = ImageEnhance.Color(masked_image)
    
    strategies = [
        (lambda: br_enhancer.enhance(.5)),
        (lambda: br_enhancer.enhance(1.5)),
        (lambda: co_enhancer.enhance(.5)),
        (lambda: co_enhancer.enhance(1.5)),
        (lambda: sa_enhancer.enhance(.5)),
        (lambda: sa_enhancer.enhance(1.5)),
    ]
    
    for i, strategy in enumerate(strategies):
        manipulated_masked_image = strategy()
        manipulated_masked_image.show()
        original_image = image.copy()
        original_image.paste(masked_image, (0, 0), mask)
        
        original_image.save(os.path.join('test_color_transfer',f'Image18_{i}.jpg'))

if __name__ == "__main__":
    image = Image.open(os.path.join('test_color_transfer','Image18.jpg')).convert('RGB')
    mask = Image.open(os.path.join('test_color_transfer','Image18_mask.jpg')).convert("L") 
    
    apply_color_transfer(image, mask)