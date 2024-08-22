import cv2
import numpy as np
import random
from PIL import Image,ImageEnhance
import os


def apply_color_transfer(image, mask):
    manipulated_images = []
    
    masked_image = Image.new("RGB", image.size, (0, 0, 0))
    masked_image.paste(image, (0, 0), mask)
    
    br_enhancer = ImageEnhance.Brightness(masked_image)
    co_enhancer = ImageEnhance.Contrast(masked_image)
    #sa_enhancer = ImageEnhance.Color(masked_image)
    
    #TODO: choose better strategies
    strategies = [
        (lambda: br_enhancer.enhance(.8)),
        (lambda: br_enhancer.enhance(.9)),
        (lambda: br_enhancer.enhance(1.1)),
        (lambda: br_enhancer.enhance(1.2)),
        (lambda: co_enhancer.enhance(.8)),
        (lambda: co_enhancer.enhance(.9)),
        (lambda: co_enhancer.enhance(1.1)),
        (lambda: co_enhancer.enhance(1.2)),
    ]
    
    for strategy in strategies:
        manipulated_masked_image = strategy()
        
        manipulated_image = image.copy()
        manipulated_image.paste(manipulated_masked_image, (0, 0), mask)
        
        manipulated_images.append(manipulated_image)
        
    return manipulated_images
    