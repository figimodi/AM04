import os
import cv2
import numpy as np
import random
import pickle
import argparse
from shutil import copyfile
from utils.spattering import generate_images_with_random_proliferation

def process_defects_masks(base_path, min_dist, max_points, max_spread, darkest_gray, lightest_gray):
    defects_masks_path = os.path.join(base_path, 'DefectsMasks')
    defects_path = os.path.join(base_path, 'Defects')
    
    proliferated_images = []

    for folder_name in os.listdir(defects_masks_path):
        folder_path = os.path.join(defects_masks_path, folder_name)

        if os.path.isdir(folder_path):
            combined_mask_01 = None
            combined_mask_02 = None

            for image_name in os.listdir(folder_path):
                # Splitting the filename to extract parts
                parts = image_name.split('.')[0].split('_')

                if len(parts) >= 3:
                    A, B, C = parts[:3]
                    D = parts[3] if len(parts) > 3 else None

                    image_path = os.path.join(folder_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        continue

                    if combined_mask_01 is None:
                        combined_mask_01 = np.zeros_like(image, dtype=np.uint8)
                    if combined_mask_02 is None:
                        combined_mask_02 = np.zeros_like(image, dtype=np.uint8)

                    # Threshold to ensure binary mask (black or white)
                    _, image_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

                    if B == 'PD' and (D is None or 'Spattering' not in D):
                        # Combine the masks for the first task
                        if combined_mask_01 is None:
                            combined_mask_01 = image_binary
                        else:
                            combined_mask_01 = cv2.bitwise_or(combined_mask_01, image_binary)

                    if D == 'Spattering':
                        # Generate and proliferate for Spattering
                        gray_image_rgb, new_mask_binary = generate_images_with_random_proliferation(
                            image_binary, min_dist=min_dist, max_points=max_points, max_spread=max_spread, darkest_gray=darkest_gray, lightest_gray=lightest_gray
                        )

                        # Save the proliferated mask in DefectsMasks as A_PDG_C_D.png
                        spattering_mask_save_path = os.path.join(folder_path, f"{A}_PDG_{C}_{D}.png")
                        cv2.imwrite(spattering_mask_save_path, new_mask_binary)
                        print(f"Saved proliferated mask: {spattering_mask_save_path}")

                        # Save the RGB image with proliferated points in Defects as A_C_Spattering.png
                        defects_save_folder = os.path.join(defects_path, A)
                        os.makedirs(defects_save_folder, exist_ok=True)
                        spattering_rgb_save_path = os.path.join(defects_save_folder, f"{A}_{C}_Spattering.png")
                        cv2.imwrite(spattering_rgb_save_path, gray_image_rgb)
                        print(f"Saved proliferated RGB points image: {spattering_rgb_save_path}")

                        proliferated_images.append(spattering_rgb_save_path)

                        # Combine the new mask into the second task combined mask
                        if combined_mask_02 is None:
                            combined_mask_02 = new_mask_binary
                        else:
                            combined_mask_02 = cv2.bitwise_or(combined_mask_02, new_mask_binary)

            # Save the combined mask for Task 1 as A_Mask_01.png
            if combined_mask_01 is not None:
                mask_01_save_path = os.path.join(folder_path, f"{folder_name}_Mask_01.png")
                cv2.imwrite(mask_01_save_path, combined_mask_01)
                print(f"Saved combined mask (Task 1): {mask_01_save_path}")

            # Re-run Task 1 but including newly generated PDG_C_Spattering masks for A_Mask_02.png
            if combined_mask_02 is not None:
                combined_mask_02 = cv2.bitwise_or(combined_mask_02, combined_mask_01)
                mask_02_save_path = os.path.join(folder_path, f"{folder_name}_Mask_02.png")
                cv2.imwrite(mask_02_save_path, combined_mask_02)
                print(f"Saved combined mask with proliferations (Task 3): {mask_02_save_path}")

    return proliferated_images

def substitute_proliferated_images(base_path, proliferated_images, num_images_to_generate, num_spattering=2):
    defects_path = os.path.join(base_path, 'Defects')
    no_defects_path = os.path.join(base_path, 'NoDefects')
    defects_masks_path = os.path.join(base_path, 'DefectsMasks')
    
    # Ensure Defects path exists
    os.makedirs(defects_path, exist_ok=True)

    # Fetch the list of no-defects images
    no_defects_images = [os.path.join(no_defects_path, img) for img in os.listdir(no_defects_path)]
    random.shuffle(no_defects_images)
    
    # Limit the selection to the specified number of images to generate
    selected_no_defects_images = no_defects_images[:min(num_images_to_generate, len(no_defects_images))]

    # Ensure there are enough proliferated images
    if len(proliferated_images) < num_spattering:
        raise ValueError("Not enough proliferated images to perform the desired spattering.")

    # Start saving from Image50
    start_index = 50

    for i, no_defect_image_path in enumerate(selected_no_defects_images):
        # Load the no-defect image
        no_defect_image = cv2.imread(no_defect_image_path, cv2.IMREAD_COLOR)
        if no_defect_image is None:
            print(f"Failed to load no-defect image: {no_defect_image_path}")
            continue

        # Initialize the image with no defects
        no_defect_with_proliferation = no_defect_image.copy()

        # initialize combined Mask
        combined_mask = cv2.cvtColor(np.zeros_like(no_defect_image, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

        # Randomly sample proliferated images
        sampled_proliferated_images = random.sample(proliferated_images, num_spattering)

        for proliferated_image_path in sampled_proliferated_images:
            # Load the proliferated image
            proliferated_image = cv2.imread(proliferated_image_path, cv2.IMREAD_UNCHANGED)
            if proliferated_image is None:
                print(f"Failed to load proliferated image: {proliferated_image_path}")
                continue

            #Load relative Mask
            parts = proliferated_image_path.split('\\')[-1].split('.')[0].split('_')
            A, C, D = parts[:3]
            proliferated_points_mask_path = os.path.join(defects_masks_path, A, f'{A}_PDG_{C}_{D}.png')
            mask = cv2.imread(proliferated_points_mask_path, cv2.IMREAD_GRAYSCALE)
            _, proliferated_points_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            print(proliferated_points_mask.shape, combined_mask.shape)
            combined_mask = cv2.bitwise_or(combined_mask, proliferated_points_mask)

            # Apply mask to get the proliferated points
            proliferated_points = cv2.bitwise_and(proliferated_image, proliferated_image, mask=proliferated_points_mask)

            # Resize proliferated points to match no-defect image size
            if proliferated_points.shape[:2] != no_defect_image.shape[:2]:
                proliferated_points = cv2.resize(proliferated_points, (no_defect_image.shape[1], no_defect_image.shape[0]))

            # Create a mask for the proliferated points in the resized image
            proliferated_points_mask = cv2.cvtColor(proliferated_points, cv2.COLOR_BGR2GRAY) > 0

            # Substitute the proliferated points onto the no-defect image
            no_defect_with_proliferation[proliferated_points_mask] = proliferated_points[proliferated_points_mask]
        

        # Save the new image in the Defects folder starting from Image50
        new_image_defects_folder = os.path.join(defects_path, f"Image{start_index + i}")
        new_image_mask_folder = os.path.join(defects_masks_path, f"Image{start_index + i}")
        os.makedirs(new_image_defects_folder, exist_ok=True)
        os.makedirs(new_image_mask_folder, exist_ok=True)

        # Get the base name of the no-defect image but change extension to jpg
        new_image_path = os.path.join(new_image_defects_folder, f"Image{start_index + i}.jpg")
        cv2.imwrite(new_image_path, no_defect_with_proliferation)
        print(f"Saved new defect image: {new_image_path}")

        mask_save_path = os.path.join(new_image_mask_folder, f"Image{start_index + i}_Mask_01.png")
        cv2.imwrite(mask_save_path, combined_mask)
        mask_save_path = os.path.join(new_image_mask_folder, f"Image{start_index + i}_Mask_02.png")
        cv2.imwrite(mask_save_path, combined_mask)
        print(f"Saved combined mask (Task 1): {mask_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--min_dist', type=int, help='Minimum distance between points', dest='MIN_DIST', default=0)    
    parser.add_argument('--max_points', type=int, help='Max number of points in the area', dest='MAX_POINTS', default=400)    
    parser.add_argument('--max_spread', type=int, help='Maximum spread among points', dest='MAX_SPREAD', default=10)    
    parser.add_argument('--darkest_gray', type=int, help='The hexadecimal value for the darkest gray', dest='DARKEST_GRAY', default=160)    
    parser.add_argument('--lightest_gray', type=int, help='The hexadecimal value for the lightest gray', dest='LIGHTEST_GRAY', default=170)    
    args = parser.parse_args()

    base_path = '../data'  # Replace with your base directory path
    
    # First, process the defect masks and generate proliferated images
    proliferated_images = process_defects_masks(base_path, args.MIN_DIST, args.MAX_POINTS, args.MAX_SPREAD, args.DARKEST_GRAY, args.LIGHTEST_GRAY)

    # Now, use some of those proliferated images to create new defect images
    num_images_to_generate = 10  # Specify the number of images to generate
    substitute_proliferated_images(base_path, proliferated_images, num_images_to_generate)

    for dir in os.listdir('../data/Defects'): 
        for file in os.listdir(os.path.join('../data/Defects', dir)): 
            if 'Spattering' in file: 
                os.remove(os.path.join('../data/Defects', dir, file))
