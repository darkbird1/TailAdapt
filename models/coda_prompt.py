import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from backbone.vit_coda_promtpt import CodaPromptSVPWrapper

from utils.panda import ClipFeatureAlignment, log_class_distribution, update_distribution, BalancedSoftmaxLoss

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        # 1. Initialize Standard Network
        self._network = CodaPromptVitNet(args, True)
        
        
        print("Injecting SVP (Sparse Variational Prompts)...")
        if self.args.get("use_svp", True): # Assume True or set in args
            # Extract params from existing config
            emb_d = 768 # Standard ViT-B dimension
            pool_size = 100 #args["pool_size"]
            prompt_len = 8 #args["prompt_len"]
            n_tasks = 10#args["n_tasks"]
            
            # Create Wrapper
            svp_prompt = CodaPromptSVPWrapper(emb_d, n_tasks, [pool_size, prompt_len])
            
            # Replace in network
            if isinstance(self._network, nn.DataParallel):
                self._network.module.prompt = svp_prompt
            else:
                self._network.prompt = svp_prompt
                
            

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.inc = args["increment"]

        self.replace_augment = args["replace_augment"]
        self.prev_task_dist = []
        self.beta = 0.3
        
        self.prev_full_class_counts: Counter = Counter()
        
        
        self.adaptive_beta = args.get("adaptive_beta", True)  
        self.beta_strategy = args.get("beta_strategy", "task_progress")  
        
        self.aug_type = 4 #1-mixup, 2-cutmix, 3-remix, 4-concutmix
        
        
        self.alignment_model = None
        self.task_accuracies = []  

        # Freeze the parameters for ViT if specified
        if self.args.get("freeze", False):
            
        
            
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad) + sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')

        
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)

        
        if hasattr(self._network, "module"):
            p_mod = self._network.module.prompt
        else:
            p_mod = self._network.prompt
            
        if hasattr(p_mod, "svp_module"):
            p_mod.svp_module.update_prior()
        # =========================================================================

        self._update_for_next_task() 

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args.get("reinit_optimizer", False):
            optimizer = self.get_optimizer()

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                params, 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
       
        pass

    
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

        # Update performance history for performance-based beta adaptation
        if self.alignment_model and self.adaptive_beta and self.beta_strategy == "performance_based":
            overall_accuracy, _ = self._compute_accuracy(self._network, self.test_loader, self._cur_task)
            self.alignment_model.update_performance(overall_accuracy / 100.0)
            logging.info(f"Task {self._cur_task}: Performance for beta adaptation: {overall_accuracy:.2f}%")

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        #accumulate metrics across epochs
        task_correct = Counter()
        task_total = Counter()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        if self.replace_augment:
            #required modules for cropAug
            distribution_vector = np.zeros(self._total_classes, dtype='float')
            distribution_vector, class_counts = log_class_distribution(train_loader, distribution_vector)
            #clip model for alignment - dist vec, device
            if self._cur_task == 0:
                self.alignment_model = ClipFeatureAlignment(
                    self._cur_task,
                    class_counts, None, None, 
                    beta=self.beta,
                    device=self._device,
                    adaptive_beta=self.adaptive_beta,
                    beta_strategy=self.beta_strategy,
                    prev_class_counts=self.prev_full_class_counts
                )
                self.alignment_model.to(self._device)
            else:
                prev_min = self.prev_task_dist[self._cur_task-1][0]
                prev_max = self.prev_task_dist[self._cur_task-1][1]
                self.alignment_model = ClipFeatureAlignment(
                    self._cur_task,
                    class_counts, prev_min, prev_max, 
                    beta=self.beta,
                    device=self._device,
                    adaptive_beta=self.adaptive_beta,
                    beta_strategy=self.beta_strategy,
                    prev_class_counts=self.prev_full_class_counts
                )
                self.alignment_model.to(self._device)

            # Log adaptive beta information
            current_beta = self.alignment_model.get_current_beta()
            logging.info(f"Task {self._cur_task}: Using adaptive beta = {current_beta:.3f} (base: {self.beta:.3f}, strategy: {self.beta_strategy})")

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.replace_augment:
                    inputs, targets, mins, maxs = self.alignment_model(inputs, targets, self._cur_task)
                    if self._cur_task == 0 or (len(self.prev_task_dist) >= self._cur_task and self.prev_task_dist[self._cur_task - 1] != [mins, maxs]):
                        self.prev_task_dist.append([mins, maxs])
                    if isinstance(targets, torch.Tensor):
                        # targets is already a tensor from ClipFeatureAlignment
                        pass  
                    else:
                        # targets is still a list, convert it
                        targets = torch.stack(targets)  
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                # logits - CodaPrompt returns logits and prompt_loss
                logits, prompt_loss = self._network(inputs, train=True)
                logits = logits[:, :self._total_classes]

                # Mask previous classes during training
                logits[:, :self._known_classes] = float('-inf')
                
                
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                
                loss = loss_supervised + prompt_loss.sum()

                
                if self.args.get("pull_constraint", False) and hasattr(self._network, 'get_reduce_sim'):
                    reduce_sim = self._network.get_reduce_sim()
                    if reduce_sim is not None:
                        loss = loss - self.args["pull_constraint_coeff"] * reduce_sim

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                #Track class wise accuracy
                for label, prediction in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                    task_correct[label] += (label == prediction)
                    task_total[label] += 1

            if scheduler:
                scheduler.step()

            #accumulate task values for wandb log
            total_loss += losses
            total_correct += correct
            total_samples += total
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc, class_wise_test_acc = self._compute_accuracy(self._network, test_loader, self._cur_task)
            
                
                print("Overall Accuracy: ", test_acc)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, class wise test acc {}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                    class_wise_test_acc,
                )
                prog_bar.set_description(info)
                logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)[:, :self._total_classes]  # CodaPrompt returns tuple
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
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
                outputs = model(inputs)[:, :self._total_classes]  # CodaPrompt returns tuple
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
            
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicts.cpu().numpy())
            
            
            for label in range(self._total_classes):
                class_correct[label] += ((predicts.cpu() == label) & (targets == label)).sum().item()
                class_total[label] += (targets == label).sum().item()
        
        # Compute per-class accuracies as percentages
        class_accuracies = {
            label: (class_correct[label] / class_total[label]) * 100 if class_total[label] > 0 else 0.0
            for label in range(self._total_classes)
        }
        
        overall_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        
        # -----------------------------------------------------------------------------
        
        return overall_acc, class_accuracies