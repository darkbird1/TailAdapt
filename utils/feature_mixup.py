import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt


class ResNetFeatureMixUp(nn.Module):
    """
    For each (head→tail) class pair, finds one head + one tail image in the batch,
    extracts intermediate feature maps from ResNet-50, performs equal-weight MixUp 
    at feature level (F' = 0.5*F_head + 0.5*F_tail), then forwards through remaining 
    network layers → yields one new tail-labeled sample.
    """
    
    def __init__(self,
                 current_task_num: int,
                 class_counts: Counter,
                 prev_min: int,
                 prev_max: int,
                 beta: float,
                 device: torch.device,
                 mixup_layer: str = "layer3",  # Options: layer1, layer2, layer3, layer4
                 adaptive_beta: bool = True,
                 beta_strategy: str = "task_progress",
                 prev_class_counts: Counter = None,
                 augmentation_strength: float = 0.3,
                 use_prompt_ensemble: bool = False,
                 save_viz: bool = False):
        super().__init__()
        self.device = device
        
        # --- ResNet-50 backbone ---
        self.resnet = models.resnet50(pretrained=True).to(device)
        self.resnet.eval()
        
        # Remove the final classification layer for feature extraction
        self.resnet.fc = nn.Identity()
        
        # Define which layer to extract features from
        self.mixup_layer = mixup_layer
        self.feature_hooks = {}
        self._register_hooks()
        
        # --- Image preprocessing for ResNet ---
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # --- Adaptive-beta machinery (unchanged) ---
        self.class_counts = class_counts
        self.base_beta = beta
        self.beta = beta
        self.adaptive_beta = adaptive_beta
        self.beta_strategy = beta_strategy
        self.distribution_changes = []
        self.performance_history = []
        self.prev_class_counts = prev_class_counts
        self.prior_min_samples = prev_min
        self.prior_max_samples = prev_max
        
        # --- Build head→tail mapping & limits ---
        mc = class_counts.most_common()
        half = len(mc) // 2
        highest = [c for c, _ in mc[:half]]
        lowest = [c for c, _ in mc[-half:]]
        self.replacement_mapping = dict(zip(highest, lowest))
        self._compute_task_level_limits()
        
        # Track current task
        self.current_task = current_task_num
        self.previous_task = None
        self.min_samples = float('inf')
        self.max_samples = float('-inf')
        
        # Visualizations
        self.save_viz = save_viz
        
        # --- Augment transforms for final image ---
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
        ])
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate feature maps."""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_hooks[name] = output
            return hook
        
        # Register hook based on specified layer
        if self.mixup_layer == "layer1":
            self.resnet.layer1.register_forward_hook(get_activation('features'))
        elif self.mixup_layer == "layer2":
            self.resnet.layer2.register_forward_hook(get_activation('features'))
        elif self.mixup_layer == "layer3":
            self.resnet.layer3.register_forward_hook(get_activation('features'))
        elif self.mixup_layer == "layer4":
            self.resnet.layer4.register_forward_hook(get_activation('features'))
        else:
            raise ValueError(f"Invalid mixup_layer: {self.mixup_layer}")
    
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
    
    def _extract_features_at_layer(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features from the specified ResNet layer."""
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = self.preprocess(image).unsqueeze(0)
            else:
                raise TypeError(f"Expected torch.Tensor or PIL.Image, got {type(image)}")
        
        # Ensure image is properly preprocessed for ResNet
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            # If not preprocessed, assume it's a raw tensor and preprocess
            if isinstance(image, torch.Tensor) and image.max() <= 1.0:
                # Convert to PIL and preprocess
                image = transforms.ToPILImage()(image.cpu())
                image = self.preprocess(image).unsqueeze(0)
        
        image = image.to(self.device)
        
        # Clear previous hooks
        self.feature_hooks.clear()
        
        # Forward pass through ResNet up to specified layer
        with torch.no_grad():
            if self.mixup_layer == "layer1":
                x = self.resnet.conv1(image)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)
                x = self.resnet.layer1(x)
            elif self.mixup_layer == "layer2":
                x = self.resnet.conv1(image)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)
                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
            elif self.mixup_layer == "layer3":
                x = self.resnet.conv1(image)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)
                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
            elif self.mixup_layer == "layer4":
                x = self.resnet.conv1(image)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)
                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)
        
        return self.feature_hooks['features']
    
    def _forward_from_features(self, mixed_features: torch.Tensor) -> torch.Tensor:
        """Forward mixed features through remaining ResNet layers to get final image."""
        with torch.no_grad():
            if self.mixup_layer == "layer1":
                x = self.resnet.layer2(mixed_features)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)
            elif self.mixup_layer == "layer2":
                x = self.resnet.layer3(mixed_features)
                x = self.resnet.layer4(x)
            elif self.mixup_layer == "layer3":
                x = self.resnet.layer4(mixed_features)
            elif self.mixup_layer == "layer4":
                # For layer4, we need to reconstruct image from high-level features
                # This is non-trivial and would require a decoder network
                # For now, we'll use adaptive average pooling and reshape
                x = F.adaptive_avg_pool2d(mixed_features, (7, 7))
            
            # Convert features back to image space (this is a simplified approximation)
            # In practice, you might want to use a trained decoder or GAN
            if x.dim() == 4:  # [B, C, H, W]
                # Simple upsampling to image size
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                # Reduce channels to 3 for RGB
                if x.shape[1] > 3:
                    x = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1), (224, 224))
                # Normalize to [0, 1]
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        return x
    
    def _create_feature_mixup(self, head_img, head_lbl, tail_img):
        """
        Extract features from head and tail images, perform equal-weight MixUp,
        and forward through remaining network to reconstruct image.
        """
        # Extract features from both images
        head_features = self._extract_features_at_layer(head_img)
        tail_features = self._extract_features_at_layer(tail_img)
        
        # Perform equal-weight MixUp: F' = 0.5 * F_head + 0.5 * F_tail
        mixed_features = 0.5 * head_features + 0.5 * tail_features
        
        # Forward through remaining layers to get reconstructed image
        reconstructed_img = self._forward_from_features(mixed_features)
        
        # Convert to PIL for augmentation
        if isinstance(reconstructed_img, torch.Tensor):
            reconstructed_img = reconstructed_img.squeeze(0).cpu()
            reconstructed_pil = transforms.ToPILImage()(reconstructed_img)
        
        # Apply augmentations
        augmented_pil = self.augment_transforms(reconstructed_pil)
        
        # New label is the tail class
        new_lbl = self.replacement_mapping[head_lbl]
        
        return augmented_pil, new_lbl
    
    def save_visualizations(self, img1, img2, mixed_img, save_dir="feature_mixup_views", basename="comparison"):
        """
        img1, img2, mixed_img: either PIL.Images or torch.Tensors (C×H×W, normalized or not)
        save_dir: folder where to write the file
        basename: filename (without extension)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        def to_numpy(img):
            if isinstance(img, torch.Tensor):
                # Unnormalize if needed
                if img.max() <= 1.0 and img.min() >= 0.0:
                    arr = img.clamp(0, 1).permute(1, 2, 0).numpy()
                else:
                    # Assume ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                    img = img.cpu() * std + mean
                    arr = img.clamp(0, 1).permute(1, 2, 0).numpy()
            elif isinstance(img, Image.Image):
                arr = np.array(img).astype(float) / 255.0
            else:
                raise TypeError(f"Unsupported image type {type(img)}")
            return arr
        
        imgs = [img1, img2, mixed_img]
        titles = ["Head Image", "Tail Image", f"Feature MixUp ({self.mixup_layer})"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, title in zip(axes, imgs, titles):
            ax.imshow(to_numpy(im))
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{basename}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved feature MixUp comparison to {out_path}")
    
    def forward(self,
                images: torch.Tensor,
                labels,
                task_num: int,
                force_beta_update: bool = True,
                copy_add_flag: bool = True,
                viz_dir: str = "feature_mixup_views"):
        
        # Convert labels to plain ints
        if torch.is_tensor(labels):
            labels = labels.cpu().tolist()
        else:
            labels = [int(l) for l in labels]
        
        # Collect tail images
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
                    # Keep head original
                    processed_images.append(img_tensor)
                    processed_labels.append(lab)
                
                if cands:
                    tail_img = cands[self.replacement_counts[lab] % len(cands)]
                    aug_pil, aug_lbl = self._create_feature_mixup(img, lab, tail_img)
                    
                    # Save 3-panel visualization
                    if self.save_viz:
                        basename = f"t{task_num}_h{lab}_t{tail_lbl}_{self.replacement_counts[lab]}_mixup_{self.mixup_layer}"
                        self.save_visualizations(img, tail_img, aug_pil, viz_dir, basename)
                    
                    aug_tensor = to_tensor(aug_pil)
                    processed_images.append(aug_tensor)
                    processed_labels.append(aug_lbl)
                else:
                    # If no tail candidate, at least keep original if replace-mode
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