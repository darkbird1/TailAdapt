import numpy as np
import pandas as pd
import os
import glob
import json
import random

#set the seed for reproducibility
np.random.seed(1993)
random.seed(1993)

#iNaturalist-------------------------------------------------

root = '/data/iNaturalist/train_val_images/'
def count_sort_folders(base_folder, output_file):
    try:
        folder_counts = []
        folder_index = 0
        for root, dirs, files in os.walk(root):
            if files:
                folder_name = os.path.basename(root)
                folder_counts.append((folder_name, len(files)))
        folder_counts.sort(key=lambda x:x[1], reverse=True)

        with open('/home/PANDA/utils/output.txt', 'w') as file:
                for index, (folder_name, file_count) in enumerate(folder_counts, start=0):
                    file.write(f"Folder {index}: {folder_name} - {file_count} files\n")
                
        print(f"File counts written")
    except Exception as e:
        print(f"An error occured {e}")

def create_inaturalist_json(root_folder, class_count=None):
    train_file = "/home/PANDA/data/iNaturalist/train_"+str(class_count)+".json"
    val_file = "/home/PANDA/data/iNaturalist/val_"+str(class_count)+".json"
    train_data = {"data_path": [], "labels": [], "label_names":[]}
    val_data = {"data_path": [], "labels": [], "label_names":[]}
    try:
        folder_counts = []
        folders_to_use = []
        folder_index = 0
        for root, dirs, files in os.walk(root_folder):
            # print(f"Root: {root}")
            # print(f"Directories: {dirs}")
            if files:
                # print(f"Files: {files}")
                
                folder_name = os.path.basename(root)
                # print(f"Folder inside : {folder_name}")
                # print(f"Temp: {root}")
                folder_counts.append((root, len(files)))
        folder_counts.sort(key=lambda x:x[1], reverse=True)
        
        #Note - Overall structure in Aves
        # Max ~ 3900 images, min ~ 16 images
        # We dont use all the 964 categories. Instead we choose two sets of categories - 300 classes and 500 classes
        if class_count == 100:
            folders_to_use = random.sample(folder_counts[65:], 100)
            for i, (folder_names, count) in enumerate(folders_to_use):
                folder_path = os.path.join(root_folder, folder_names)
                all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                random.shuffle(all_images)
                val_images = all_images[-8:]
                train_images = all_images[:-8]

                for img in train_images:
                    train_data["data_path"].append(os.path.join(folder_path, img))
                    train_data["labels"].append(i)
                    train_data["label_names"].append(folder_path.split("/")[6])
                for img in val_images:
                    val_data["data_path"].append(os.path.join(folder_path, img))
                    val_data["labels"].append(i)
                    val_data["label_names"].append(folder_path.split("/")[6])
            
            with open(train_file, "w") as f:
                json.dump(train_data, f, indent=4)
            with open(val_file, "w") as f:
                json.dump(val_data, f, indent=4)

            print(f"Train and validation JSON files created successfully for class count = {class_count}")

        elif class_count == 300:
            folders_to_use = folder_counts[100:400]
            for i, (folder_names, count) in enumerate(folders_to_use):
                folder_path = os.path.join(root_folder, folder_names)
                all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                random.shuffle(all_images)
                val_images = all_images[-25:]
                train_images = all_images[:-25]

                for img in train_images:
                    train_data["data_path"].append(os.path.join(folder_path, img))
                    train_data["labels"].append(i)
                    train_data["label_names"].append(folder_path.split("/")[6])
                for img in val_images:
                    val_data["data_path"].append(os.path.join(folder_path, img))
                    val_data["labels"].append(i)
                    val_data["label_names"].append(folder_path.split("/")[6])
            
            with open(train_file, "w") as f:
                json.dump(train_data, f, indent=4)
            with open(val_file, "w") as f:
                json.dump(val_data, f, indent=4)

            print(f"Train and validation JSON files created successfully for class count = {class_count}")

        elif class_count == 500:
            folders_to_use = random.sample(folder_counts[65:], 500)
            for i, (folder_names, count) in enumerate(folders_to_use):
                folder_path = os.path.join(root_folder, folder_names)
                all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                random.shuffle(all_images)
                val_images = all_images[-8:]
                train_images = all_images[:-8]

                for img in train_images:
                    train_data["data_path"].append(os.path.join(folder_path, img))
                    train_data["labels"].append(i)
                    train_data["label_names"].append(folder_path.split("/")[6])
                for img in val_images:
                    val_data["data_path"].append(os.path.join(folder_path, img))
                    val_data["labels"].append(i)
                    val_data["label_names"].append(folder_path.split("/")[6])
            
            with open(train_file, "w") as f:
                json.dump(train_data, f, indent=4)
            with open(val_file, "w") as f:
                json.dump(val_data, f, indent=4)

            print(f"Train and validation JSON files created successfully for class count = {class_count}")

        elif class_count == 800:
            folders_to_use = folder_counts[-800:] #get the last 800 categories
            for i, (folder_names, count) in enumerate(folders_to_use):
                folder_path = os.path.join(root_folder, folder_names)
                all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                random.shuffle(all_images)
                val_images = all_images[-10:] #only 10 validation images as some images only have 15 samples
                train_images = all_images[:-10]

                for img in train_images:
                    train_data["data_path"].append(os.path.join(folder_path, img))
                    train_data["labels"].append(i)
                    train_data["label_names"].append(folder_path.split("/")[6])
                for img in val_images:
                    val_data["data_path"].append(os.path.join(folder_path, img))
                    val_data["labels"].append(i)
                    val_data["label_names"].append(folder_path.split("/")[6])
            
            with open(train_file, "w") as f:
                json.dump(train_data, f, indent=4)
            with open(val_file, "w") as f:
                json.dump(val_data, f, indent=4)

            print(f"Train and validation JSON files created successfully for class count = {class_count}")
        
        if class_count == 5089:
            folders_to_use = folder_counts[-5089:] #get the last 800 categories
            for i, (folder_names, count) in enumerate(folders_to_use):
                folder_path = os.path.join(root_folder, folder_names)
                all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                random.shuffle(all_images)
                val_images = all_images[-10:] #only 10 validation images as some images only have 15 samples
                train_images = all_images[:-10]

                for img in train_images:
                    train_data["data_path"].append(os.path.join(folder_path, img))
                    train_data["labels"].append(i)
                    train_data["label_names"].append(folder_path.split("/")[6])
                for img in val_images:
                    val_data["data_path"].append(os.path.join(folder_path, img))
                    val_data["labels"].append(i)
                    val_data["label_names"].append(folder_path.split("/")[6])
            
            with open(train_file, "w") as f:
                json.dump(train_data, f, indent=4)
            with open(val_file, "w") as f:
                json.dump(val_data, f, indent=4)

            print(f"Train and validation JSON files created successfully for class count = {class_count}")
            
        # elif class_count is None:
        #     # case of mixed datasets



    except Exception as e:
        print(f"Encountered an error: {e}")


# create_inaturalist_json(root, 100)

def inaturalist_label_create(train_file, out_file):
    with open(train_file, "r") as input:
        data_json = json.load(input)
    data_paths = data_json["data_path"]
    labels = data_json["labels"]
    labels_set = set(data_json["labels"])

    out_dict = {"labels":[], "class_name":[]}
    for i in labels_set:
        class_name = data_paths[labels.index(i)].split("/")[6]
        out_dict["labels"].append(i)
        out_dict["class_name"].append(class_name)
    
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=4)
    # print(out_dict)

#Food101-------------------------------------------------
train_path = "/home/PANDA/data/food101/train_file.json"
train_path_upd = "/home/PANDA/data/food101/train_file_upd.json"
test_path = "/home/PANDA/data/food101/test_file.json"
test_path_upd = "/home/PANDA/data/food101/test_file_upd.json"
val_path = "/home/PANDA/data/food101/valid_file.json"
val_path_upd = "/home/PANDA/data/food101/valid_file_upd.json"

class_names = "/home/PANDA/data/food101/labels.txt"


def food_101_json_maker(org_path, updated_path):
    label_name_dict = {}
    with open(class_names, "r") as files:
        for index, row in enumerate(files):
            label_name_dict[index] = row.strip()

    with open(org_path, "r") as file:
        json_data = json.load(file)

    json_data["class_name"] = []
    for entry in json_data["labels"]:
        if entry is not None:
            json_data["class_name"].append(label_name_dict[entry])

    with open(updated_path, "w") as file:
        json.dump(json_data, file, indent=4)

food_101_json_maker(val_path, val_path_upd)