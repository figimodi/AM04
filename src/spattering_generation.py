import numpy as np
import cv2
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def generate_random_points(mask, min_dist, max_points=100):
    """
    Generate random points in the white areas of a binary mask.
    Points will be at least min_dist apart.
    
    Args:
    - mask: Binary mask where white areas are valid for point placement.
    - min_dist: Minimum distance between any two points.
    - max_points: Maximum number of points to place.
    
    Returns:
    - points: List of (x, y) tuples representing the points.
    """
    # Find white pixel coordinates
    white_coords = np.column_stack(np.where(mask == 255))
    
    points = []
    attempts = 0
    
    while len(points) < max_points and attempts < max_points * 10:
        # Randomly choose a white pixel
        idx = np.random.randint(0, len(white_coords))
        candidate_point = white_coords[idx]
        
        if len(points) == 0:
            points.append(candidate_point)
        else:
            # Ensure the new point is at least min_dist away from all existing points
            distances = cdist([candidate_point], points)
            if np.all(distances >= min_dist):
                points.append(candidate_point)
        
        attempts += 1
    
    return points

def proliferate_points_randomly(image, points, max_spread=10, darkest_gray=129, lightest_gray=196, newOrigin_Threshold: float = 0.7):
    """
    Proliferate points in an image randomly, using RGB format.
    
    Args:
    - image: The image where points are to be proliferated (RGB format).
    - points: List of (x, y) tuples representing the points.
    - max_spread: Maximum number of times a point can proliferate.
    - darkest_gray: Darkest gray level (0-255) for the points.
    - lightest_gray: Lightest gray level (0-255) for the points.
    
    Returns:
    - image: The updated RGB image with proliferated points.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 possible directions
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
                    if random.random() > newOrigin_Threshold:
                        x, y = new_x, new_y  # Move the spread origin to this new point

    return image

def generate_images_with_random_proliferation(mask, min_dist=10, max_points=100, max_spread=10, darkest_gray=129, lightest_gray=196):
    """
    Generate two images based on the input mask: one with random RGB proliferation and another binary image.
    
    Args:
    - mask: Binary mask where white areas are valid for point placement.
    - min_dist: Minimum distance between points.
    - max_points: Maximum number of points to generate.
    - max_spread: Maximum number of times a point can proliferate.
    - darkest_gray: Darkest gray level (0-255) for the points.
    - lightest_gray: Lightest gray level (0-255) for the points.
    
    Returns:
    - gray_image_rgb: The image with gray points in RGB and random proliferation.
    - binary_image: The binary image where proliferation areas are white and the rest is black.
    """
    # Generate random points on the white areas of the mask
    points = generate_random_points(mask, min_dist, max_points)
    
    # Create an empty RGB image for proliferation
    gray_image_rgb = np.ones((*mask.shape, 3), dtype=np.uint8)*255
    
    # Proliferate points on the RGB image randomly
    gray_image_rgb = proliferate_points_randomly(gray_image_rgb, points, max_spread, darkest_gray, lightest_gray)
    
    # Create a binary image from the RGB image
    binary_image = np.any(gray_image_rgb < 255, axis=-1).astype(np.uint8) * 255
    
    return gray_image_rgb, binary_image

if __name__ == '__main__':
    # Load the mask from a PNG file
    mask_path = 'data\DefectsMasks\Image23\Image23_Mask_0.jpg'  # Replace with your PNG file path
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the mask is binary (i.e., only 0 or 255 values)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Generate images with the given parameters
    gray_image_rgb, binary_image = generate_images_with_random_proliferation(
        mask, min_dist=22, max_points=100, max_spread=25, darkest_gray=126, lightest_gray=200
    )

    rgb_path = 'data\Spattering\Spattering.png'
    binary_path = 'data\Spattering\Mask.png'

    cv2.imwrite(rgb_path, gray_image_rgb)
    cv2.imwrite(binary_path, binary_image)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Original Mask Image')

    plt.subplot(1, 3, 2)
    plt.imshow(gray_image_rgb)
    plt.title('RGB Image with Random Proliferation')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image from Proliferation')

    plt.show()
