import os
import zipfile
import re
import sys


def unzip_and_rename(zip_folder_path):
    for root, dirs, files in os.walk(zip_folder_path):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                folder_name = os.path.splitext(file)[0]
                extracted_folder = os.path.join(root, folder_name)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder)

                rename_files_in_folder(extracted_folder, folder_name)

def rename_files_in_folder(folder_path, folder_name):
    pd_counter = 1
    
    for root, _, files in os.walk(folder_path):
        print(f' Extracting and renaming {root}')
        for file in files:
            if file.endswith('.txt') or file.endswith('.png'):
                os.remove(os.path.join(root, file))
                continue
            
            parts = file.split(',')
            if len(parts) < 4:
                continue

            defectType = parts[3]
            if defectType.startswith('Background'):
                os.remove(os.path.join(root, file))
                continue

            if defectType.startswith('Mask'):
                B = 'Mask'
                C = '0'
                D = ''
            elif defectType.startswith('Laser'):
                B = 'L'
                C = '0'
                D = ''
            else:
                B = 'PD'
                C = f'{pd_counter:02}'
                pd_counter += 1
                D = re.sub(r'\d', '', defectType).strip()
            
            new_name = f"{folder_name}_{B}_{C}{'_' + D if D else ''}.jpg"
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, new_name)
            
            os.rename(old_path, new_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_zip_folders>")
        sys.exit(1)
    
    zip_folder_path = sys.argv[1]
    unzip_and_rename(zip_folder_path)
