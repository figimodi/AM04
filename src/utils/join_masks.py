import os
import sys
import cv2

def process_defects_masks(base_path):
    defects_masks_path = os.path.join(base_path, 'DefectsMasks')

    for folder_name in os.listdir(defects_masks_path):
        folder_path = os.path.join(defects_masks_path, folder_name)

        if os.path.isdir(folder_path):
            combined_mask = None

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

                    # Threshold to ensure binary mask (black or white)
                    _, image_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

                    if B == 'PD' and (D is None or 'Spattering' not in D):
                        # Combine the masks for the first task
                        if combined_mask is None:
                            combined_mask = image_binary
                        else:
                            combined_mask = cv2.bitwise_or(combined_mask, image_binary)

            # Save the combined mask for Task 1 as A_Mask_01.png
            if combined_mask is not None:
                mask_01_save_path = os.path.join(defects_masks_path, f"{folder_name}_Mask_00.png")
                cv2.imwrite(mask_01_save_path, combined_mask)
                print(f"Saved combined mask (Task 1): {mask_01_save_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python join_mask.py <base_path>")
        sys.exit(1)
    base_path = sys.argv[1]
    
    process_defects_masks(base_path)