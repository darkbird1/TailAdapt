import logging
import numpy as np
import math
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import random
from collections import Counter
import clip
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os





def update_distribution(dist, current_data):
    
    for data in current_data:
        temp_label = int(data)
        dist[int(temp_label)] += 1
    
    # print('dist vec: ',dist)
    return dist


def log_class_distribution(dataloaders, dist_vec):
        class_counts = Counter()
        for _, _, targets in dataloaders:
            class_counts.update(targets.cpu().numpy())
        # print(f"Class counts: {class_counts}")
        # class_distribution = {int(clss):count for clss, count in class_counts.items()}
        for index, value in class_counts.items():
            dist_vec[index] = value
        # print(f"Class distribution: {dist_vec}")
        return dist_vec, class_counts



class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, device):
        super().__init__()
        # numerator = (class_size / total_tasks) * (which_task+1)
        cls_prior = cls_num_list / sum(cls_num_list)
        cls_prior = torch.FloatTensor(cls_prior).to(device)
        self.log_prior = torch.log(cls_prior).unsqueeze(0)

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior
        # print('min and max target balanced softmax: ', labels.min(), labels.max())
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss

##############################################################
############### ADAPTIVE BETA VALUE ##########################
###############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from torchvision import transforms
from collections import Counter
import numpy as np
import random

import os, time, random
import numpy as np
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import transforms

class ClipFeatureAlignment(nn.Module):
    """
    For each (head→tail) class pair, finds one head + one tail image in the batch,
    extracts top-50% CLIP-salient patches from each (resized to 224×224),
    swaps them into the head image, reconstructs + augments → yields one new tail-labeled sample.
    """
    def __init__(self,
                 current_task_num: int,
                 class_counts: Counter,
                 prev_min: int,
                 prev_max: int,
                 beta: float,
                 device: torch.device,
                 num_patches: int = 4,
                 adaptive_beta: bool = True,
                 beta_strategy: str = "task_progress",
                 prev_class_counts: Counter = None,
                 augmentation_strength: float = 0.3,
                 patch_replacement_ratio: float = 0.5,
                 use_prompt_ensemble: bool = False,
                 save_viz: bool = False):
        super().__init__()
        self.device = device

        # --- CLIP backbone & preprocess ---
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.clip_model.eval()

        # --- Grid / patch settings ---
        self.num_patches = num_patches
        self.patch_size = 224 // num_patches
        self.patch_replacement_ratio = patch_replacement_ratio

        # --- Adaptive-beta machinery (unchanged) ---
        self.class_counts = class_counts
        self.base_beta = beta
        self.beta = beta
        self.adaptive_beta = adaptive_beta
        self.beta_strategy = beta_strategy
        self.distribution_changes = []
        self.performance_history   = []
        self.prev_class_counts     = prev_class_counts
        self.prior_min_samples     = prev_min
        self.prior_max_samples     = prev_max

        # --- Build head→tail mapping & limits ---
        mc = class_counts.most_common()
        half = len(mc) // 2
        highest = [c for c,_ in mc[:half]]
        lowest  = [c for c,_ in mc[-half:]]
        self.replacement_mapping = dict(zip(highest, lowest))
        self._compute_task_level_limits()

        # track current task
        self.current_task  = current_task_num
        self.previous_task = None
        self.min_samples   = float('inf')
        self.max_samples   = float('-inf')

        #visualizations
        self.save_viz = save_viz
        # --- Augment transforms for final image ---
        self.augment_transforms = transforms.Compose([
            # transforms.ColorJitter(
            #     brightness=augmentation_strength * 0.4,
            #     contrast=augmentation_strength * 0.4,
            #     saturation=augmentation_strength * 0.4,
            #     hue=augmentation_strength * 0.1
            # ),
            # transforms.RandomAffine(
            #     degrees=augmentation_strength * 15,
            #     translate=(augmentation_strength * 0.1, augmentation_strength * 0.1),
            #     scale=(1 - augmentation_strength * 0.1, 1 + augmentation_strength * 0.1)
            # ),
            transforms.RandomHorizontalFlip(p=0.3),
        ])

    def get_current_beta(self):
        """Get the current beta value being used."""
        return self.beta
    
    def _compute_task_level_limits(self):
        """Compute replacement mapping and adjust limits based on prior task min/max samples."""
        highest_tuples = self.class_counts.most_common(3)  
        lowest_tuples = list(self.class_counts.most_common()[:-4:-1])  

        highest_classes = [t[0] for t in highest_tuples]
        lowest_classes = [t[0] for t in lowest_tuples]

        self.replacement_mapping = {high: low for high, low in zip(highest_classes, lowest_classes)}
        self.replacement_limits = {}

        current_min = min(self.class_counts.values(), default=0)
        current_max = max(self.class_counts.values(), default=0)

        if self.prior_min_samples is not None and self.prior_max_samples is not None:
            adjusted_min = self.beta * self.prior_min_samples + (1 - self.beta) * current_min
            adjusted_max = self.beta * self.prior_max_samples + (1 - self.beta) * current_max
        else:
            adjusted_min, adjusted_max = current_min, current_max

        updated_class_counts = self.class_counts.copy()

        for (high, high_count), (low, low_count) in zip(highest_tuples, lowest_tuples):
            adjusted_limit = min(low_count, (adjusted_max - adjusted_min) // 2)
            self.replacement_limits[high] = adjusted_limit

            moved_samples = min(self.replacement_limits[high], high_count)  
            updated_class_counts[high] -= moved_samples  
            updated_class_counts[low] += moved_samples  

        self.min_samples = min(updated_class_counts.values())
        self.max_samples = max(updated_class_counts.values())
        self.replacement_counts = {high: 0 for high in self.replacement_mapping.keys()}
        
    def _extract_patches_from_image(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image.cpu())
        image = image.resize((224, 224))
        t = self.preprocess(image).unsqueeze(0)  # [1,C,224,224]
        patches = F.unfold(t, kernel_size=self.patch_size, stride=self.patch_size)
        P = self.num_patches ** 2
        C = t.shape[1]
        patches = patches.squeeze(0).transpose(0,1).view(P, C, self.patch_size, self.patch_size)
        return patches  # [P, C, patch_size, patch_size]

    def _extract_patches_raw(self, image: Image.Image) -> torch.Tensor:
        """Split a 224×224 PIL into P= num_patches² raw patches in [0,1]."""
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image.cpu())
        image = image.resize((224,224))
        raw = transforms.ToTensor()(image).to(self.device).unsqueeze(0)  # [1,C,224,224]
        patches = F.unfold(raw, kernel_size=self.patch_size, stride=self.patch_size)
        P = self.num_patches**2
        C = raw.shape[1]
        patches = patches.squeeze(0).transpose(0,1).view(P, C, self.patch_size, self.patch_size)
        return patches  # [P,C,h,w]
    
    def _reconstruct_image_from_patches(self, patches: torch.Tensor) -> torch.Tensor:
        P, C, h, w = patches.shape
        flat = patches.view(P, C*h*w).transpose(0,1).unsqueeze(0)
        recon = F.fold(flat, output_size=(224,224), kernel_size=h, stride=h)
        return recon.squeeze(0)  # [C,224,224]

    def extract_features_for_image(self, image, label):
        """
        Split into patches, resize each to 224×224, then encode with CLIP and compute
        cosine similarity to the text prompt "a photo of a {label}".
        """
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image.cpu())
        image = image.resize((224,224))
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)  # [1,C,224,224]

        # unfold into patches
        patches = F.unfold(img_t, kernel_size=self.patch_size, stride=self.patch_size)
        P = patches.shape[-1]
        C = img_t.shape[1]
        patches = patches.squeeze(0).transpose(0,1).view(P, C, self.patch_size, self.patch_size)

        # upsample each patch back to 224×224
        patches = F.interpolate(patches,
                                 size=(224,224),
                                 mode='bilinear',
                                 align_corners=False)

        # encode patches and prompt
        with torch.no_grad():
            patch_emb = self.clip_model.encode_image(patches)  # [P, D]
            txt = clip.tokenize([f"a photo of a {label}"]).to(self.device)
            txt_emb = self.clip_model.encode_text(txt).squeeze(0)  # [D]
            sims = F.cosine_similarity(patch_emb, txt_emb.unsqueeze(0), dim=1)  # [P]
        return sims.cpu()

    def _create_augmented_copy_with_patch_replacement(self, head_img, head_lbl, tail_img):
        # ensure PIL
        if isinstance(head_img, torch.Tensor):
            head_img = transforms.ToPILImage()(head_img.cpu())
        if isinstance(tail_img, torch.Tensor):
            tail_img = transforms.ToPILImage()(tail_img.cpu())

        head_p = self._extract_patches_raw(head_img)
        tail_p = self._extract_patches_raw(tail_img)
        P = head_p.shape[0]
        k = max(1, int(P * self.patch_replacement_ratio))

        # score & pick top-k
        h_s = self.extract_features_for_image(head_img, head_lbl)
        t_s = self.extract_features_for_image(tail_img, self.replacement_mapping[head_lbl])
        hi = torch.topk(h_s, k).indices.tolist()
        ti = torch.topk(t_s, k).indices.tolist()

        # swap patches
        mod = head_p.clone().to(self.device)
        for h_i, t_i in zip(hi, ti):
            mod[h_i] = tail_p[t_i].to(self.device)

        # reconstruct & augment
        recon_t = self._reconstruct_image_from_patches(mod)
        recon_p = transforms.ToPILImage()(recon_t.cpu())
        aug_p   = self.augment_transforms(recon_p)

        # new label is the tail class
        new_lbl = self.replacement_mapping[head_lbl]
        return aug_p, new_lbl

    def save_visualizations(self, img1, img2, patched_img, save_dir="patch_views", basename="comparison"):
        """
        img1, img2, patched_img: either PIL.Images or torch.Tensors (C×H×W, normalized or not)
        save_dir: folder where to write the file
        basename: filename (without extension)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        def to_numpy(img):
            # If tensor, move to CPU and convert to H×W×C in [0,1]
            if isinstance(img, torch.Tensor):
                # if normalized, unnormalize here (example means/std — adjust if needed)
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:,None,None]
                std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:,None,None]
                img = img.cpu() #* std + mean  
                arr = img.clamp(0,1).permute(1,2,0).numpy()
            elif isinstance(img, Image.Image):
                arr = np.array(img).astype(float)/255.0
            else:
                raise TypeError(f"Unsupported image type {type(img)}")
            return arr

        imgs = [img1, img2, patched_img]
        titles = ["Input 1", "Input 2", "Patched / Augmented"]

        # --- Option A: single figure with 3 panels ---
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        for ax, im, title in zip(axes, imgs, titles):
            ax.imshow(to_numpy(im))
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{basename}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved comparison figure to {out_path}")

    def forward(self,
                images: torch.Tensor,
                labels,
                task_num: int,
                force_beta_update: bool = True,
                copy_add_flag: bool = True,
                viz_dir: str = "patch_views"):
        # convert labels to plain ints
        if torch.is_tensor(labels):
            labels = labels.cpu().tolist()
        else:
            labels = [int(l) for l in labels]

        # … (adaptive-beta and limits code unchanged) …

        # collect tail images
        tail_map = {}
        for img, lbl in zip(images, labels):
            if lbl in self.replacement_mapping.values():
                tail_map.setdefault(lbl, []).append(img)

        processed_images, processed_labels = [], []
        to_tensor = transforms.ToTensor()
        os.makedirs(viz_dir, exist_ok=True)

        for img, lab in zip(images, labels):
            img_tensor = img.cpu() if isinstance(img, torch.Tensor) else to_tensor(img)
            is_head = lab in self.replacement_mapping and \
                      self.replacement_counts.get(lab, 0) < self.replacement_limits.get(lab, 0)

            if is_head:
                tail_lbl = self.replacement_mapping[lab]
                cands = tail_map.get(tail_lbl, [])

                if copy_add_flag:
                    # keep head original
                    processed_images.append(img_tensor)
                    processed_labels.append(lab)

                if cands:
                    tail_img = cands[self.replacement_counts[lab] % len(cands)]
                    aug_pil, aug_lbl = self._create_augmented_copy_with_patch_replacement(
                        img, lab, tail_img
                    )

                    # save 3-panel viz
                    if self.save_viz:
                        basename = f"t{task_num}_h{lab}_t{tail_lbl}_{self.replacement_counts[lab]}"
                        self.save_visualizations(img, tail_img, aug_pil, viz_dir, basename)

                    aug_tensor = to_tensor(aug_pil)
                    processed_images.append(aug_tensor)
                    processed_labels.append(aug_lbl)
                else:
                    # if no tail candidate, at least keep original if replace-mode
                    if not copy_add_flag:
                        processed_images.append(img_tensor)
                        processed_labels.append(lab)

                self.replacement_counts[lab] += 1
            else:
                processed_images.append(img_tensor)
                processed_labels.append(lab)

        batch_images = torch.stack(processed_images).to(self.device)
        counts = Counter(processed_labels)
        self.min_samples = min(counts.values())
        self.max_samples = max(counts.values())

        return batch_images, processed_labels, self.min_samples, self.max_samples
    
####################################################################################
##################### -- ATTN MASK METHOD --  ##################### ################
####################################################################################
# import os
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import clip
# from PIL import Image
# from collections import Counter
# from scipy.stats import wasserstein_distance
# import matplotlib.pyplot as plt
# from torchvision import transforms

# def save_head_tail_attn_patch(head_img, tail_img, attn_map, patched_img,
#                               save_dir, basename="comp"):
#     """
#     Save a 1×4 figure: [ head | tail | attention | patched ].
#     head_img, tail_img, patched_img: PIL or torch.Tensor [C,H,W] in [0,1]
#     attn_map:        1D numpy array length P = num_patches**2
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     def to_np(img):
#         if isinstance(img, torch.Tensor):
#             arr = img.cpu().permute(1,2,0).numpy()
#         elif isinstance(img, Image.Image):
#             arr = np.array(img).astype(np.float32)/255.0
#         else:
#             raise TypeError(type(img))
#         return np.clip(arr, 0, 1)

#     head_np    = to_np(head_img)
#     tail_np    = to_np(tail_img)
#     patched_np = to_np(patched_img)

#     attn = np.array(attn_map, copy=False)
#     if attn.ndim == 1:
#         n = int(math.sqrt(attn.size))
#         attn = attn.reshape(n, n)
#     attn = (attn - attn.min())/(attn.max()-attn.min()+1e-8)

#     fig, axes = plt.subplots(1,4, figsize=(16,4))
#     axes[0].imshow(head_np);    axes[0].set_title("Head");    axes[0].axis("off")
#     axes[1].imshow(tail_np);    axes[1].set_title("Tail");    axes[1].axis("off")
#     im = axes[2].imshow(attn, cmap="hot", vmin=0, vmax=1)
#     axes[2].set_title("Attention"); axes[2].axis("off")
#     plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
#     axes[3].imshow(patched_np); axes[3].set_title("Patched"); axes[3].axis("off")

#     plt.tight_layout()
#     out_path = os.path.join(save_dir, f"{basename}.png")
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved 4-panel comparison → {out_path}")


# class ClipFeatureAlignmentAttn(nn.Module):
#     def __init__(self,
#                  current_task_num: int,
#                  class_counts: Counter,
#                  prev_min: int,
#                  prev_max: int,
#                  beta: float,
#                  device: torch.device,
#                  num_patches: int = 4,
#                  adaptive_beta: bool = True,
#                  beta_strategy: str = "task_progress",
#                  prev_class_counts: Counter = None):
#         super().__init__()
#         self.device = device
#         self.beta = beta
#         self.base_beta = beta
#         self.adaptive_beta = adaptive_beta
#         self.beta_strategy = beta_strategy
#         self.class_counts = class_counts
#         self.prior_min_samples = prev_min
#         self.prior_max_samples = prev_max
#         self.current_task = current_task_num
#         self.prev_class_counts = prev_class_counts or Counter()
#         self.num_patches = num_patches
#         self.patch_size = 224 // num_patches
#         self.patch_replacement_ratio = 0.5

#         # load CLIP
#         self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device)
#         self.clip_model.eval()
#         self.vit = self.clip_model.visual
#         self.clip_dtype = next(self.vit.parameters()).dtype

#         # compute head→tail mapping & limits
#         self._compute_task_level_limits()
#         self.min_samples = min(class_counts.values()) if class_counts else 0
#         self.max_samples = max(class_counts.values()) if class_counts else 0

#     def get_current_beta(self):
#         """Get the current beta value being used."""
#         return self.beta
    
#     def _compute_task_level_limits(self):
#         """Compute replacement mapping and adjust limits based on prior task min/max samples."""
#         highest_tuples = self.class_counts.most_common(3)  
#         lowest_tuples = list(self.class_counts.most_common()[:-4:-1])  

#         highest_classes = [t[0] for t in highest_tuples]
#         lowest_classes = [t[0] for t in lowest_tuples]

#         self.replacement_mapping = {high: low for high, low in zip(highest_classes, lowest_classes)}
#         self.replacement_limits = {}

#         current_min = min(self.class_counts.values(), default=0)
#         current_max = max(self.class_counts.values(), default=0)

#         if self.prior_min_samples is not None and self.prior_max_samples is not None:
#             adjusted_min = self.beta * self.prior_min_samples + (1 - self.beta) * current_min
#             adjusted_max = self.beta * self.prior_max_samples + (1 - self.beta) * current_max
#         else:
#             adjusted_min, adjusted_max = current_min, current_max

#         updated_class_counts = self.class_counts.copy()

#         for (high, high_count), (low, low_count) in zip(highest_tuples, lowest_tuples):
#             adjusted_limit = min(low_count, (adjusted_max - adjusted_min) // 2)
#             self.replacement_limits[high] = adjusted_limit

#             moved_samples = min(self.replacement_limits[high], high_count)  
#             updated_class_counts[high] -= moved_samples  
#             updated_class_counts[low] += moved_samples  

#         self.min_samples = min(updated_class_counts.values())
#         self.max_samples = max(updated_class_counts.values())
#         self.replacement_counts = {high: 0 for high in self.replacement_mapping.keys()}        

#     def forward_with_attn(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Run the ViT forward and capture each block's attention map.
#         Properly handles PyTorch MultiheadAttention modules.
#         Returns: CLS→patch affinities [B, num_tokens-1].
#         """
#         B = x.shape[0]

#         conv1_dtype = self.vit.conv1.weight.dtype
#         x = x.to(device=self.device, dtype=conv1_dtype)
#         # Ensure input matches CLIP model dtype
#         # x = x.to(dtype=self.clip_dtype)

#         # (a) patchify
#         x = self.vit.conv1(x)                                   # [B, D,14,14]
#         x = x.view(B, x.shape[1], -1).permute(0,2,1)           # [B,196,D]

#         # (b) prepend CLS + add pos‐emb
#         cls_tok = self.vit.class_embedding.to(x.dtype)          # [D] or [1,D]
#         # Ensure cls_tok has the right shape: [1, 1, D]
#         if cls_tok.dim() == 1:
#             cls_tok = cls_tok.unsqueeze(0).unsqueeze(0)          # [D] -> [1,1,D]
#         elif cls_tok.dim() == 2:
#             cls_tok = cls_tok.unsqueeze(0)                       # [1,D] -> [1,1,D]
#         cls_tok = cls_tok.expand(B, -1, -1)                     # [B,1,D]
#         x = torch.cat([cls_tok, x], dim=1)                     # [B,197,D]
#         x = x + self.vit.positional_embedding.to(x.dtype)

#         # (c) run Transformer blocks & gather attn
#         attn_maps = []
        
#         for i, blk in enumerate(self.vit.transformer.resblocks):
#             y = blk.ln_1(x)
            
#             # For PyTorch MultiheadAttention, we need to:
#             # 1. Transpose from [B, N, D] to [N, B, D] (seq_len first)
#             # 2. Call with need_weights=True to get attention weights
#             # 3. Transpose back
            
#             y_transposed = y.transpose(0, 1)  # [197, B, D]
            
#             try:
#                 # PyTorch MultiheadAttention expects (seq_len, batch, embed_dim)
#                 attn_output, attn_weights = blk.attn(
#                     query=y_transposed,
#                     key=y_transposed, 
#                     value=y_transposed,
#                     need_weights=True,
#                     average_attn_weights=True  # Average over heads
#                 )
                
#                 # Transpose back to [B, N, D]
#                 attn_output = attn_output.transpose(0, 1)
#                 x = x + attn_output
                
#                 # attn_weights shape: [B, N, N] where N=197
#                 if attn_weights is not None:
#                     attn_maps.append(attn_weights)
#                     # if i == 0:  # Only print for first block to avoid spam
#                     #     print(f"Block {i}: Successfully extracted attention with shape {attn_weights.shape}")
#                 else:
#                     if i == 0:
#                         print(f"Block {i}: Attention weights returned as None")
                    
#             except Exception as e:
#                 # if i == 0:  # Only print for first block
#                 #     print(f"Block {i}: Attention extraction failed: {e}")
#                 # Fallback: just do the forward pass without getting weights
#                 try:
#                     attn_output, _ = blk.attn(
#                         query=y_transposed,
#                         key=y_transposed,
#                         value=y_transposed,
#                         need_weights=False
#                     )
#                     attn_output = attn_output.transpose(0, 1)
#                     x = x + attn_output
#                 except Exception as e2:
#                     if i == 0:
#                         print(f"Block {i}: Even fallback failed: {e2}")
#                     x = x + y  # Skip connection
            
#             # Continue with the rest of the transformer block
#             x = x + blk.mlp(blk.ln_2(x))

#         # (d) Process attention maps
#         if attn_maps:
#             # print(f"Successfully extracted attention from {len(attn_maps)} layers")
            
#             # Stack and average over layers: [num_layers, B, 197, 197] -> [B, 197, 197]
#             stacked_attn = torch.stack(attn_maps, dim=0)
#             avg_attn = stacked_attn.mean(dim=0)  # Average over layers
            
#             # Extract CLS to patch attention (skip CLS token at index 0)
#             cls_to_patch = avg_attn[:, 0, 1:]  # [B, 196]
            
#             # Verify we have actual attention (not uniform)
#             attention_std = cls_to_patch.std(dim=1).mean().item()
#             # print(f"Attention standard deviation: {attention_std:.6f}")
            
#             if attention_std < 1e-6:
#                 print("WARNING: Attention appears to be uniform (very low std)")
            
#             return cls_to_patch
#         else:
#             print("ERROR: Could not extract any attention weights - using uniform attention")
#             uniform_attn = torch.ones(B, 196).to(x.device) / 196
#             return uniform_attn

#     def _extract_patches_raw(self, image: Image.Image) -> torch.Tensor:
#         """Split a 224×224 PIL into P= num_patches² raw patches in [0,1]."""
#         if not isinstance(image, Image.Image):
#             image = transforms.ToPILImage()(image.cpu())
#         image = image.resize((224,224))
#         raw = transforms.ToTensor()(image).to(self.device).unsqueeze(0)  # [1,C,224,224]
#         patches = F.unfold(raw, kernel_size=self.patch_size, stride=self.patch_size)
#         P = self.num_patches**2
#         C = raw.shape[1]
#         patches = patches.squeeze(0).transpose(0,1).view(P, C, self.patch_size, self.patch_size)
#         return patches  # [P,C,h,w]
    
    
#     def _reconstruct_image_from_patches(self, patches: torch.Tensor) -> torch.Tensor:
#         P, C, h, w = patches.shape
#         flat = patches.view(P, C*h*w).transpose(0,1).unsqueeze(0)
#         recon = F.fold(flat, output_size=(224,224), kernel_size=h, stride=h)
#         return recon.squeeze(0)  # [C,224,224]
    
#     def extract_features_for_image(self, image, label):
#         """
#         Split into patches, resize each to 224×224, then encode with CLIP and compute
#         cosine similarity to the text prompt "a photo of a {label}".
#         """
#         if not isinstance(image, Image.Image):
#             image = transforms.ToPILImage()(image.cpu())
#         image = image.resize((224,224))
#         img_t = self.preprocess(image).unsqueeze(0).to(self.device)  # [1,C,224,224]

#         # unfold into patches
#         patches = F.unfold(img_t, kernel_size=self.patch_size, stride=self.patch_size)
#         P = patches.shape[-1]
#         C = img_t.shape[1]
#         patches = patches.squeeze(0).transpose(0,1).view(P, C, self.patch_size, self.patch_size)

#         # upsample each patch back to 224×224
#         patches = F.interpolate(patches,
#                                  size=(224,224),
#                                  mode='bilinear',
#                                  align_corners=False)

#         # encode patches and prompt
#         with torch.no_grad():
#             patch_emb = self.clip_model.encode_image(patches)  # [P, D]
#             txt = clip.tokenize([f"a photo of a {label}"]).to(self.device)
#             txt_emb = self.clip_model.encode_text(txt).squeeze(0)  # [D]
#             sims = F.cosine_similarity(patch_emb, txt_emb.unsqueeze(0), dim=1)  # [P]
#         return sims.cpu()

#     def _create_augmented_copy_with_patch_replacement(self, head_img, head_lbl, tail_img):
#         # ensure PIL
#         if isinstance(head_img, torch.Tensor):
#             head_img = transforms.ToPILImage()(head_img.cpu())
#         if isinstance(tail_img, torch.Tensor):
#             tail_img = transforms.ToPILImage()(tail_img.cpu())

#         head_p = self._extract_patches_raw(head_img)
#         tail_p = self._extract_patches_raw(tail_img)
#         P = head_p.shape[0]
#         k = max(1, int(P * self.patch_replacement_ratio))

#         # score & pick top-k
#         h_s = self.extract_features_for_image(head_img, head_lbl)
#         t_s = self.extract_features_for_image(tail_img, self.replacement_mapping[head_lbl])
#         hi = torch.topk(h_s, k).indices.tolist()
#         ti = torch.topk(t_s, k).indices.tolist()

#         # swap patches
#         mod = head_p.clone().to(self.device)
#         for h_i, t_i in zip(hi, ti):
#             mod[h_i] = tail_p[t_i].to(self.device)

#         # reconstruct & augment
#         recon_t = self._reconstruct_image_from_patches(mod)
#         recon_p = transforms.ToPILImage()(recon_t.cpu())
#         # aug_p   = self.augment_transforms(recon_p)

#         # new label is the tail class
#         new_lbl = self.replacement_mapping[head_lbl]
#         return recon_p, new_lbl
    
#     def forward(self, images, labels, task_num):
#         # on new task
#         if task_num!=self.current_task:
#             self.current_task = task_num
#             self._compute_task_level_limits()

#         # prepare batch
#         proc = []
#         for img in images:
#             if not isinstance(img, torch.Tensor):
#                 img = img if isinstance(img,Image.Image) else transforms.ToPILImage()(img)
#                 img = img.resize((224,224))
#                 img = self.preprocess(img)
#             else:
#                 if img.shape[1:]!=(224,224):
#                     img = F.interpolate(img.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
#             proc.append(img)
#         batch = torch.stack(proc).to(self.device)

#         # get attention affinities
#         cls2patch = self.forward_with_attn(batch).cpu()  # [B,196]
#         aff = cls2patch.view(-1,14,14).unsqueeze(1)
#         aff_small = F.adaptive_avg_pool2d(aff, (self.num_patches,self.num_patches))
#         affinities = aff_small.view(batch.size(0), -1)  # [B,P]

#         # convert labels
#         labs = [int(l) if torch.is_tensor(l) else l for l in labels]

#         processed_images, processed_labels = [], []
#         to_tensor = transforms.ToTensor()
#         viz_dir="patch_views"
#         os.makedirs(viz_dir, exist_ok=True)

#         # build tail pools
#         tail_map = {}
#         for i,l in enumerate(labs):
#             if l in self.replacement_mapping.values():
#                 tail_map.setdefault(l, []).append(images[i])

#         # loop
#         for i,(orig_img,l) in enumerate(zip(images,labs)):
#             is_head = l in self.replacement_mapping and \
#                       self.replacement_counts[l] < self.replacement_limits[l]
#             # always keep original
#             img_tensor = batch[i].cpu()
#             processed_images.append(img_tensor)
#             processed_labels.append(l)
#             if is_head:
#                 tail_lbl = self.replacement_mapping[l]
#                 cands = tail_map.get(tail_lbl, [])
#                 if cands:
#                     tail_img = cands[self.replacement_counts[l] % len(cands)]
#                     patched_pil, new_lbl = self._create_augmented_copy_with_patch_replacement(
#                     orig_img, l, tail_img
#                 )
#                     # save comparison
#                     save_head_tail_attn_patch(
#                         orig_img, tail_img,
#                         affinities[i].detach().numpy(), patched_pil, 
#                         save_dir="attn_views",
#                         basename=f"task{task_num}_h{l}_t{tail_lbl}_{self.replacement_counts[l]}"
#                     )
#                     # patch-swap
#                     aug_pil, new_lbl = self._create_augmented_copy_with_patch_replacement(
#                         img, l, tail_img
#                     )
#                     processed_images.append(to_tensor(aug_pil))
#                     processed_labels.append(new_lbl)
#                     self.replacement_counts[l] += 1

#         batch_images = torch.stack(processed_images).to(self.device)
#         counts = Counter(processed_labels)
#         self.min_samples = min(counts.values())
#         self.max_samples = max(counts.values())

#         return batch_images, processed_labels, self.min_samples, self.max_samples
