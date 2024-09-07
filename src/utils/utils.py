import cv2
import numpy as np
import random
import os
from PIL import Image, ImageEnhance
from scipy.spatial.distance import cdist


def apply_color_transfer(image, mask):
    manipulated_images = []
    
    masked_image = Image.new("RGB", image.size, (0, 0, 0))
    masked_image.paste(image, (0, 0), mask)
    
    br_enhancer = ImageEnhance.Brightness(masked_image)
    co_enhancer = ImageEnhance.Contrast(masked_image)
    #sa_enhancer = ImageEnhance.Color(masked_image)
    
    #TODO: choose better strategies
    strategies = [
        (lambda: co_enhancer.enhance(.8)),
        (lambda: co_enhancer.enhance(.85)),
        (lambda: co_enhancer.enhance(.9)),
        (lambda: co_enhancer.enhance(.95)),
        (lambda: co_enhancer.enhance(1.05)),
        (lambda: co_enhancer.enhance(1.1)),
        (lambda: co_enhancer.enhance(1.15)),
        (lambda: co_enhancer.enhance(1.2)),
    ]
    
    for strategy in strategies:
        manipulated_masked_image = strategy()
        
        manipulated_image = image.copy()
        manipulated_image.paste(manipulated_masked_image, (0, 0), mask)
        
        manipulated_images.append(manipulated_image)
        
    return manipulated_images
    
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

def recurive_rename(folder):
    for element in folder:
        if os.path.isfile(element):
            os.rename(element, os.path.splitext(element)[0] + '.png')
        elif os.path.isdir(element):
            recurive_rename(element)

def generate_random_points(min_dist, max_points): 
    min_c, max_c = 500, 1040
    min_r, max_r = 330, 700
    
    max_points = random.randint(max_points-100, max_points+100)
    
    center = np.array([random.randint(min_r, max_r), random.randint(min_c, max_c)])
    
    points = [] 
     
    while len(points) < max_points : 
        candidate_point = int(np.random.normal(center[0], random.randint(20,50))), int(np.random.normal(center[1], random.randint(20,50)))
         
        if len(points) == 0: 
            points.append(candidate_point) 
        else: 
            distances = cdist([candidate_point], points) 
            if np.all(distances >= min_dist): 
                points.append(candidate_point) 
     
    return points

def proliferate_points_randomly(image, points, max_spread, darkest_gray, lightest_gray, new_origin_threshold: float = 0.7):

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 possible directions
    directions_weights = [0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05]
    for point in points:
        x, y = point
        tot_spread_count = random.randint(1, max_spread)
        spread_count = 0
        gray_color = (darkest_gray, darkest_gray, darkest_gray)
        image[x,y] = gray_color
        
        while  spread_count < tot_spread_count:
            # Randomly choose a direction to spread
            dx, dy = random.choices(directions,weights=directions_weights, k=1)[0]
            new_x, new_y = x + dx, y + dy
            
            if 0 <= new_x < image.shape[0] and 0 <= new_y < image.shape[1]:
                # Set a random gray intensity for the spread point
                #intensity = random.randint(darkest_gray, lightest_gray)
                gray_color = (lightest_gray, lightest_gray, lightest_gray)
                if np.all(image[new_x, new_y] == 255):
                    image[new_x, new_y] = gray_color
                    spread_count += 1
                else:
                    # Darken the pixel if it's already set
                    current_intensity = int(np.mean(image[new_x, new_y]))
                    new_intensity = random.randint(darkest_gray, current_intensity)
                    gray_color = (new_intensity, new_intensity, new_intensity)
                    image[new_x, new_y] = gray_color
                    if random.random() > new_origin_threshold:
                        x, y = new_x, new_y  # Move the spread origin to this new point

    return image

def generate_images_with_random_proliferation(min_dist, max_points, max_spread, darkest_gray, lightest_gray):
    # Generate random points on the white areas of the mask
    points = generate_random_points(min_dist, max_points)
    
    # Create an empty RGB image for proliferation
    gray_image_rgb = np.ones((1024, 1280, 3), dtype=np.uint8)*255
    
    # Proliferate points on the RGB image randomly
    gray_image_rgb = proliferate_points_randomly(gray_image_rgb, points, max_spread, darkest_gray, lightest_gray)
    
    # Create a binary image from the RGB image
    binary_image = np.any(gray_image_rgb < 255, axis=-1).astype(np.uint8) * 255
    
    return gray_image_rgb, binary_image