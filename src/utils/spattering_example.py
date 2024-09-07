import numpy as np
import cv2
import argparse
from PIL import Image
import matplotlib.pyplot as plt


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

    plt.figure(figsize=(8,8))
    nodefect_image = Image.fromarray(gray_image_rgb).convert('RGB') 
    mask = Image.fromarray(binary_image).convert('L') 
    background = Image.open('../../data/NoDefects/Image1.jpg').convert('RGB') 
    background.paste(nodefect_image, (0, 0), mask) 
    plt.imshow(background)
    plt.title('Example')

    plt.show()
