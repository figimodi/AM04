import os

defects_masks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'DefectsMasks')
defect_counts = {}

for folder in os.listdir(defects_masks_path):
    for file in os.listdir(os.path.join(defects_masks_path, folder)):
        if 'Mask' not in file:
            defect_type = file.split('_')[-1].split('.')[0]

            if defect_type in defect_counts:
                defect_counts[defect_type] += 1
            else:
                defect_counts[defect_type] = 1
        
for defect_type, count in defect_counts.items():
    print(f"{defect_type}: {count}")

total_defects = sum(defect_counts.values())
print(f"Total defects: {total_defects}")