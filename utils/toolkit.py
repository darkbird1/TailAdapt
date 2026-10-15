import os
import numpy as np
import torch
import json
from PIL import Image
import random


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy, for initial classes
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    # for incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def create_exponential_distribution(num_classes, max_samples, rho):
    min_samples = max_samples * rho
    exp_distribution = np.exp(np.linspace(0, 1, num_classes))
    # Normalize the distribution so that it ranges from min_samples to max_samples
    distribution = min_samples + (exp_distribution - exp_distribution.min()) * (max_samples - min_samples) / (exp_distribution.max() - exp_distribution.min())
    # Random shuffle the distribution - seeded for reproducibility 
    random.shuffle(distribution)
    # Convert to integer
    distribution = distribution.astype(int)
    
    return distribution.tolist()

# 
def create_dual_exp_distribution(
    num_classes,
    max_samples,
    rho_upper,
    rho_star,      # inner ratio for creating imbalance within chunk
    rho_num,       # 1-based chunk index to which we apply rho_star
    increment
):
    """
    Creates a dual exponential distribution with global exponential decay
    and localized imbalance within a specific chunk.
    
    Args:
        num_classes: Total number of classes
        max_samples: Maximum samples for the most frequent class
        rho_upper: Global imbalance ratio (min/max across all classes)
        rho_star: Inner imbalance ratio (min/max within the target chunk)
        rho_num: 1-based index of chunk to apply inner imbalance to
        increment: Size of each chunk (5 or 10)
    """
    
    # 1) Create global exponential distribution
    min_samples = int(max_samples * rho_upper)
    exp_vals = np.exp(np.linspace(0, 1, num_classes))
    dist = min_samples + (
        (exp_vals - exp_vals.min())
        * (max_samples - min_samples)
        / (exp_vals.max() - exp_vals.min())
    )
    dist = np.flip(dist).astype(int)
    
    
    # 2) Apply inner imbalance to the chosen chunk
    target_chunk = rho_num - 1  # Convert to 0-based index
    start = target_chunk * increment
    end = min(start + increment, len(dist))
    
    # Validate chunk exists
    if start >= len(dist):
        raise IndexError(f"rho_num={rho_num} out of range for {num_classes} classes with increment {increment}")
    
    # Get the chunk and find its current range
    chunk_vals = dist[start:end].copy()
    chunk_max = chunk_vals.max()
    chunk_min = chunk_vals.min()
    
    # Create exponential imbalance within the chunk
    # The new minimum should be rho_star * current_max
    new_min = max(int(chunk_max * rho_star), 1)
    
    if len(chunk_vals) > 1:
        # Create exponential decay within the chunk
        chunk_exp_vals = np.exp(np.linspace(0, 1, len(chunk_vals)))
        new_chunk = new_min + (
            (chunk_exp_vals - chunk_exp_vals.min())
            * (chunk_max - new_min)
            / (chunk_exp_vals.max() - chunk_exp_vals.min())
        )
        new_chunk = np.flip(new_chunk).astype(int)
    else:
        # Single element chunk
        new_chunk = np.array([chunk_max])
    
    # Apply the new values to the distribution
    dist[start:end] = new_chunk
    
    return dist.tolist()

def unjson(file):
#    print('current path: ', os.getcwd())
    with open(file) as json_file:
        data = json.load(json_file)
        #print('train data size: ', len(x) )
    return data

# def load_img_from_array(idx):
#     img = Image.fromarray(np.uint8(self.phase_dataset_data[idx])).convert('RGB')        
#     return img

def get_img_from_path(data_dict, dataset):
    img_data = []
    if dataset == 'food101':
        for paths in data_dict:
            temp = os.path.join('/data1/raghav12/', paths)
            if os.path.exists(temp):
                img_data.append(temp)
            else:
                print('path does not exist: ', temp)
    else: 
        for paths in data_dict:
            # print('paths: ', paths)
            if os.path.exists(paths):
                img_data.append(paths)
                # img = cv2.imread(paths)
                # if len(img.shape) == 3:
                #     # img = cv2.merge((img, img, img))
                #     
                # #print('shape of img before stack: ', np.shape(img))           
            else:
                print('path does not exist: ', paths)

    img_data = np.array(img_data) 
    #print('shape of img:',np.shape(img_data))
    return img_data
