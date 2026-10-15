import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

import wandb
import random
from collections import Counter
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.panda import log_class_distribution, update_distribution, BalancedSoftmaxLoss
from utils.panda import ClipFeatureAlignment #Attn as ClipFeatureAlignment


num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = PromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.class_size = args["num_classes"]
        self.inc = args.get("increment", 10)

        self.replace_augment = args["replace_augment"]
        self.prev_task_dist = []
        self.beta = args.get("beta", 0.5)
        
        self.prev_full_class_counts: Counter = Counter()
        self.adaptive_beta = args.get("adaptive_beta", True)
        self.beta_strategy = args.get("beta_strategy", "task_progress")
        
        self.aug_type = 4
        self.alignment_model = None
        self.task_accuracies = []

        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        
        for n, p in self._network.backbone.named_parameters():
            if 'g_prompt' in n and any(k in n for k in ['q_m', 'q_a', 'q_b', 'q_log_s']):
                p.requires_grad = True

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')

    def after_task(self):
        
        if hasattr(self._network.backbone, 'update_svp_priors'):
            logging.info("SVP: Updating G-Prompt priors...")
            self._network.backbone.update_svp_priors()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self._update_for_next_task()
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            
    def evaluate_prompt_sparsity(self, keep_ratios=[1.0, 0.25, 0.10, 0.05]):
        model = self._network.backbone
        if not hasattr(model, 'g_prompt'):
            return
    
        with torch.no_grad():
            original_q_m = model.g_prompt.q_m.clone()
    
        results = {}
    
        for r in keep_ratios:
            with torch.no_grad():
                model.g_prompt.q_m.copy_(original_q_m)
            model.g_prompt.apply_topk_mask(keep_ratio=r)
    
            acc, _ = self._compute_accuracy(
                self._network,
                self.test_loader,
                self._cur_task
            )
            results[r] = acc
            print(f"[Prompt Sparsity] Keep {int(r*100)}% ? Acc = {acc:.2f}")
    
        # restore
        with torch.no_grad():
            model.g_prompt.q_m.copy_(original_q_m)
        return results


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()
            
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        
        
        self.evaluate_prompt_sparsity(
            keep_ratios=[1.0, 0.25, 0.10, 0.05, 0.01]
        )

    def get_optimizer(self):
        
        svp_var_params = [] # q_a, q_b
        svp_weights = []    # q_m
        other_params = []
        
        for name, p in self._network.named_parameters():
            if not p.requires_grad: continue
            
            if 'g_prompt' in name and any(k in name for k in ['q_a', 'q_b', 'q_log_s']):
                svp_var_params.append(p)
            elif 'g_prompt' in name and 'q_m' in name:
                svp_weights.append(p)
            else:
                other_params.append(p)
        
        param_groups = [
            
            {'params': svp_var_params, 'lr': self.init_lr * 20.0, 'weight_decay': 0.0},
            
            {'params': svp_weights, 'lr': self.init_lr, 'weight_decay': self.weight_decay},
            
            {'params': other_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(param_groups)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(param_groups)

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler
        
    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt (DualPrompt style)
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    if model.e_prompt.prompt.grad is not None:
                        #print(f'type: {model.e_prompt.prompt.grad}')
                        model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()
                
        
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.e_prompt.prompt_key.grad.zero_()
                model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    
    def _update_for_next_task(self):
        self._known_classes = self._total_classes

        
        if self.train_dataset is not None and len(self.train_dataset) > 0:
            temp_loader_for_counts = DataLoader(
                self.train_dataset,
                batch_size=min(len(self.train_dataset), self.batch_size * 2),
                shuffle=False,
                num_workers=num_workers
            )
            distribution_vector_for_prev = np.zeros(self._total_classes, dtype='float')
            _, final_current_task_counts = log_class_distribution(
                temp_loader_for_counts,
                distribution_vector_for_prev
            )
            if not isinstance(final_current_task_counts, Counter):
                final_current_task_counts = Counter({i: count for i, count in enumerate(final_current_task_counts)})

            self.prev_full_class_counts = final_current_task_counts.copy()
            print(f'In Learner (end of task {self._cur_task}): Stored prev_full_class_counts: {self.prev_full_class_counts}')
            logging.info(f"Task {self._cur_task} finished. Storing its distribution for next task: {self.prev_full_class_counts}")
        else:
            logging.info(f"Task {self._cur_task}: No train_dataset or empty dataset found for prev_full_class_counts update.")

        
        if self.alignment_model and self.adaptive_beta and self.beta_strategy == "performance_based":
            overall_accuracy, _ = self._compute_accuracy(self._network, self.test_loader, self._cur_task)
            self.alignment_model.update_performance(overall_accuracy / 100.0)
            logging.info(f"Task {self._cur_task}: Performance for beta adaptation: {overall_accuracy:.2f}%")

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        
        
        task_correct = Counter()
        task_total = Counter()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if self.replace_augment:
            distribution_vector = np.zeros(self._total_classes, dtype='float')
            distribution_vector, class_counts = log_class_distribution(train_loader, distribution_vector)
            if self._cur_task == 0:
                self.alignment_model = ClipFeatureAlignment(self._cur_task, class_counts, None, None, beta=self.beta, device=self._device, adaptive_beta=self.adaptive_beta, beta_strategy=self.beta_strategy, prev_class_counts=self.prev_full_class_counts)
                self.alignment_model.to(self._device)
            else:
                prev_min = self.prev_task_dist[self._cur_task-1][0] if len(self.prev_task_dist) > 0 else None
                prev_max = self.prev_task_dist[self._cur_task-1][1] if len(self.prev_task_dist) > 0 else None
                self.alignment_model = ClipFeatureAlignment(self._cur_task, class_counts, prev_min, prev_max, beta=self.beta, device=self._device, adaptive_beta=self.adaptive_beta, beta_strategy=self.beta_strategy, prev_class_counts=self.prev_full_class_counts)
                self.alignment_model.to(self._device)

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            
            warmup_epochs = 5
            target_beta = 0.01
            beta_kl = ((epoch - warmup_epochs) / (self.args['tuned_epoch'] - warmup_epochs)) * target_beta if epoch >= warmup_epochs else 0.0

            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                if hasattr(self._network.backbone, 'reset_kl'):
                    self._network.backbone.reset_kl()

                if self.replace_augment:
                    inputs, targets, mins, maxs = self.alignment_model(inputs, targets, self._cur_task)
                    if self._cur_task == 0 or (len(self.prev_task_dist) >= self._cur_task and self.prev_task_dist[self._cur_task - 1] != [mins, maxs]):
                        self.prev_task_dist.append([mins, maxs])
                    if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets, dtype=torch.long, device=inputs.device)
                        
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                
                
                if hasattr(self._network.backbone, 'get_total_kl'):
                    kl_loss = self._network.backbone.get_total_kl()
                    loss += (beta_kl * kl_loss * 1e-3)

                optimizer.zero_grad()
                loss.backward()
                
               
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler: scheduler.step()

            if (epoch + 1) % 5 == 0:
                test_acc, _ = self._compute_accuracy(self._network, test_loader, self._cur_task)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args['tuned_epoch'], losses / len(train_loader), tensor2numpy(correct) * 100 / total, test_acc)
                prog_bar.set_description(info)
                logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader, cur_task):
        model.eval()
        correct, total = 0, 0
        class_correct = [0] * self._total_classes
        class_total = [0] * self._total_classes
        all_targets = []
        all_preds = []
        
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
            
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicts.cpu().numpy())
            
            # Compute per-class accuracy
            for label in range(self._total_classes):
                class_correct[label] += ((predicts.cpu() == label) & (targets == label)).sum().item()
                class_total[label] += (targets == label).sum().item()
        
        # Compute per-class accuracies as percentages
        class_accuracies = {
            label: (class_correct[label] / class_total[label]) * 100 if class_total[label] > 0 else 0.0
            for label in range(self._total_classes)
        }
        print(class_accuracies)
        overall_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        
        # Create and Save Class-wise Accuracy Heatmap
        from sklearn.metrics import confusion_matrix
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        selected_labels = list(range(self._total_classes))
        
        # Compute confusion matrix for the selected classes
        cm = confusion_matrix(all_targets, all_preds, labels=selected_labels)
        # Normalize the confusion matrix by row
        cm_normalized = (cm.astype('float') / cm.sum(axis=1, keepdims=True)) * 100
        
        # Create a DataFrame for better labeling in the heat map
        df_cm = pd.DataFrame(cm_normalized,
                            index=[f"{i}" for i in selected_labels],
                            columns=[f"{i}" for i in selected_labels])
        
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        ax = sns.heatmap(df_cm, annot=False, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Accuracy (%)'})
        plt.title("Class-wise Accuracy Heatmap (Normalized Confusion Matrix)", fontsize=16)
        plt.xlabel("Predicted Class", fontsize=14)
        plt.ylabel("True Class", fontsize=14)
        plt.tight_layout()
        name = "heatmap"+str(cur_task)+".png"
        plt.savefig(name, dpi=300, bbox_inches="tight")
        plt.close()
        
        return overall_acc, class_accuracies