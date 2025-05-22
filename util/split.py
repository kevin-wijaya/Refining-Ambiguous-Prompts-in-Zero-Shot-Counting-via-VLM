import json
import os

# Path
split_json_path = '../FSC147_384_V2/Train_Test_Val_FSC_147.json'
output_folder = '../FSC147_384_V2/'

# Load file JSON
with open(split_json_path, 'r') as f:
    splits = json.load(f)

# save file
for split_name, image_list in splits.items():
    output_path = os.path.join(output_folder, f"{split_name}.txt")
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            for image_id in image_list:
                f.write(f"{image_id}\n")
    else:
        print(f"{split_name}.txt already exists. Skipping.")


print("finish generated txt file based on json file!")