import torch
import numpy as np
import random
import torch.nn.functional as F

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, C, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    r = np.sqrt(1 - lam)
    cut_w = int(W * r)
    cut_h = int(H * r)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def remix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    if lam > 0.5:
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
    else:
        mixed_x = x
        y_a, y_b = y, y
        lam = 1.0
    return mixed_x, y_a, y_b, lam

def contrastive_cutmix_data(x, y, alpha=1.0):
    """
    Applies Contrastive CutMix augmentation by producing two augmented views.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, C, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    r = np.sqrt(1 - lam)
    cut_w = int(W * r)
    cut_h = int(H * r)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    view1 = x.clone()
    view2 = x.clone()
    view1[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    view2[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return view1, view2, y_a, y_b, lam

# ---------------- Custom Collate Function ----------------

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def custom_collate_fn(images, labels, augmentation_type=1,
                      mixup_alpha=1.0, cutmix_alpha=1.0,
                      remix_alpha=1.0, contrastive_cutmix_alpha=1.0,
                      num_classes=100):
    """
    Custom collate function that processes each mini-batch, applies one augmentation
    technique, and returns the augmented images and original labels.
    
    For contrastive cutmix, both augmented views are concatenated to form an expanded batch.
    """

    # images = torch.stack(images)
    # labels = torch.tensor(targets)
    
    if augmentation_type == 1:
        images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
        new_labels = labels
        return images, new_labels

    elif augmentation_type == 2:
        images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
        new_labels = labels
        return images, new_labels

    elif augmentation_type == 3:
        images, y_a, y_b, lam = remix_data(images, labels, alpha=remix_alpha)
        new_labels = labels
        return images, new_labels

    elif augmentation_type == 4:
        view1, view2, y_a, y_b, lam = contrastive_cutmix_data(images, labels, alpha=contrastive_cutmix_alpha)
        new_labels = labels
        # Concatenate both augmented views along the batch dimension.
        new_images = torch.cat([view1, view2], dim=0)
        # Duplicate the labels accordingly.
        new_labels = torch.cat([new_labels, new_labels], dim=0)
        return new_images, new_labels

    else:
        new_labels = labels
        return images, new_labels

def get_augmented_dataloader(dataset, batch_size=48, augmentation_type=1,
                             mixup_alpha=1.0, cutmix_alpha=1.0,
                             remix_alpha=1.0, contrastive_cutmix_alpha=1.0,
                             num_classes=100):
    """
    Constructs a dataloader that processes the dataset in mini-batches.
    The custom collate function applies augmentation to each batch separately.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        collate_fn=lambda batch: custom_collate_fn(
                            batch, 
                            augmentation_type=augmentation_type,
                            mixup_alpha=mixup_alpha,
                            cutmix_alpha=cutmix_alpha,
                            remix_alpha=remix_alpha,
                            contrastive_cutmix_alpha=contrastive_cutmix_alpha,
                            num_classes=num_classes))
    return loader


