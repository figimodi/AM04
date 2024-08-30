import numpy as np
import cv2
import random
import argparse
from scipy.spatial.distance import cdist
from PIL import Image
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--min_dist', type=int, help='Minimum distance between points', dest='MIN_DIST', default=0)    
    parser.add_argument('--max_points', type=int, help='Max number of points in the area', dest='MAX_POINTS', default=400)    
    parser.add_argument('--max_spread', type=int, help='Maximum spread among points', dest='MAX_SPREAD', default=10)    
    parser.add_argument('--darkest_gray', type=int, help='The hexadecimal value for the darkest gray', dest='DARKEST_GRAY', default=160)    
    parser.add_argument('--lightest_gray', type=int, help='The hexadecimal value for the lightest gray', dest='LIGHTEST_GRAY', default=170)    
    args = parser.parse_args()

    # Load the mask from a PNG file
    mask_path = '../../data/DefectsMasks/Image23/Image23_Mask_0.png'  # Replace with your PNG file path
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the mask is binary (i.e., only 0 or 255 values)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Generate images with the given parameters
    gray_image_rgb, binary_image = generate_images_with_random_proliferation( 
        mask, min_dist=args.MIN_DIST, max_points=args.MAX_POINTS, max_spread=args.MAX_SPREAD, darkest_gray=args.DARKEST_GRAY, lightest_gray=args.LIGHTEST_GRAY 
    )

    # rgb_path = 'data\Spattering\Spattering.png'
    # binary_path = 'data\Spattering\Mask.png'

    # cv2.imwrite(rgb_path, gray_image_rgb)
    # cv2.imwrite(binary_path, binary_image)

    # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 3, 1)
    # plt.imshow(mask, cmap='gray')
    # plt.title('Original Mask Image')

    # plt.subplot(1, 3, 2)
    # plt.imshow(gray_image_rgb)
    # plt.title('RGB Image with Random Proliferation')

    # plt.subplot(1, 3, 3)
    # plt.imshow(binary_image, cmap='gray')
    # plt.title('Binary Image from Proliferation')

    plt.figure(figsize=(8,8))
    nodefect_image = Image.fromarray(gray_image_rgb).convert('RGB') 
    mask = Image.fromarray(binary_image).convert('L') 
    background = Image.open('../../data/NoDefects/Image1.jpg').convert('RGB') 
    background.paste(nodefect_image, (0, 0), mask) 
    plt.imshow(background)
    plt.title('Example')

    plt.show()
