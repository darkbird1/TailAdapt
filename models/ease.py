import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import EaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

import wandb
import random
from collections import Counter
import pandas as pd

from utils.panda import ClipFeatureAlignment, log_class_distribution, update_distribution, BalancedSoftmaxLoss

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = EaseNet(args, True)
        
       
        for n, p in self._network.backbone.named_parameters():
            if 'q_' in n: # SVP params
                p.requires_grad = True
        
        # Move to device
        self._network.to(self._device)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]
        
        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] 
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        self.replace_augment = args["replace_augment"]
        self.prev_task_dist = []
        self.new_beta = args["beta_replace"]
        
        self.prev_full_class_counts: Counter = Counter()
        
        self.adaptive_beta = args.get("adaptive_beta", True)
        self.beta_strategy = args.get("beta_strategy", "task_progress")
        
        self.aug_type = 4 
        self.alignment_model = None
        self.task_accuracies = []  
        
        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1 
            self.beta = 1

        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} model total parameters.')

    def after_task(self):
        
        if hasattr(self._network.backbone, 'update_svp_priors'):
            print("SVP: Updating priors via moment matching...")
            self._network.backbone.update_svp_priors()
        
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        return start_cls, end_cls
        
    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()
        
        
        if self.replace_augment:
            distribution_vector = np.zeros(self._total_classes, dtype='float')
            distribution_vector, class_counts = log_class_distribution(train_loader, distribution_vector)
            
            if self._cur_task == 0:
                self.alignment_model = ClipFeatureAlignment(
                    self._cur_task, class_counts, None, None, 
                    beta=self.new_beta, device=self._device,
                    adaptive_beta=self.adaptive_beta, beta_strategy=self.beta_strategy,
                    prev_class_counts=self.prev_full_class_counts
                ).to(self._device)
            else:
                if len(self.prev_task_dist) > self._cur_task - 1:
                    prev_min = self.prev_task_dist[self._cur_task-1][0]
                    prev_max = self.prev_task_dist[self._cur_task-1][1]
                else:
                    prev_min = None; prev_max = None
                
                self.alignment_model = ClipFeatureAlignment(
                    self._cur_task, class_counts, prev_min, prev_max, 
                    beta=self.new_beta, device=self._device,
                    adaptive_beta=self.adaptive_beta, beta_strategy=self.beta_strategy,
                    prev_class_counts=self.prev_full_class_counts
                ).to(self._device)

        with torch.no_grad():           
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0
            
            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam and index > self.adapter_num - 1: break
                elif self.use_diagonal and index != -1 and index != self._cur_task: continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    if self.replace_augment:
                        data, labels, mins, maxs = self.alignment_model(data, label, self._cur_task)
                        
                        while len(self.prev_task_dist) <= self._cur_task:
                            self.prev_task_dist.append([None, None])
                        if self._cur_task == 0 or self.prev_task_dist[self._cur_task] != [mins, maxs]:
                            self.prev_task_dist[self._cur_task] = [mins, maxs]
                        
                        if isinstance(labels, torch.Tensor): label = labels  
                        elif isinstance(labels, (list, tuple)):
                            if len(labels) > 0 and isinstance(labels[0], torch.Tensor): label = torch.stack(labels)
                            else: label = torch.tensor(labels, dtype=torch.long)
                        else: label = torch.tensor(labels, dtype=torch.long)
                            
                    data = data.to(self._device)
                    label = label.to(self._device)
                    
                    
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto

            if self.use_exemplars and self._cur_task > 0:
                pass 
        
        if self.use_diagonal or self.use_exemplars: return
        
        if self.recalc_sim: self.solve_sim_reset()
        else: self.solve_similarity()
    
    
    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        start_cls, end_cls = self.get_cls_range(task_id)
        A = self._network.fc.weight.data[self._known_classes:, start_dim : end_dim]
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        A_hat = self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim]
        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self):       
        for task_id in range(self._cur_task):          
            start_cls, end_cls = self.get_cls_range(task_id=task_id)
            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)
            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)
            similarity = F.softmax(similarity, dim=1)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]
            self._network.fc.weight.data[start_cls : end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2: break
            if self.use_init_ptm: range_dim = range(task_id + 2, self._cur_task + 2)
            else: range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num: break
                start_cls, end_cls = self.get_cls_range(task_id=task_id)
                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim
                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()
                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)
                similarity = F.softmax(similarity, dim=1)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]
                self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim] = B_hat.to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        
        self._update_for_next_task() 
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            if "later_lr" not in self.args or self.args["later_lr"] == 0: self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0: self.args["later_epochs"] = self.args["init_epochs"]
            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)
    
    def get_optimizer(self, lr):
        
        svp_var_params = [] 
        svp_weights = []    # q_m
        other_params = []   
        
        for name, p in self._network.named_parameters():
            if not p.requires_grad: continue
            
            if 'q_a' in name or 'q_b' in name or 'q_log_s' in name:
                svp_var_params.append(p)
            elif 'q_m' in name:
                svp_weights.append(p)
            else:
                other_params.append(p)
        
        param_groups = [
            
            {'params': svp_var_params, 'lr': lr, 'weight_decay': 0.0},
            
            {'params': svp_weights, 'lr': lr, 'weight_decay': self.weight_decay},
            
            {'params': other_params, 'lr': lr, 'weight_decay': self.weight_decay}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(param_groups)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(param_groups)

        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.get("init_milestones", []), gamma=self.args.get("init_lr_decay", 0.1))
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler

    def _update_for_next_task(self):
        
        self._known_classes = self._total_classes
        if self.train_dataset is not None and len(self.train_dataset) > 0:
            temp_loader = DataLoader(self.train_dataset, batch_size=min(len(self.train_dataset), self.batch_size * 2), shuffle=False, num_workers=num_workers)
            dist_vec = np.zeros(self._total_classes, dtype='float')
            _, counts = log_class_distribution(temp_loader, dist_vec)
            if not isinstance(counts, Counter): counts = Counter({i: c for i, c in enumerate(counts)})
            self.prev_full_class_counts = counts.copy()
            
            while len(self.prev_task_dist) <= self._cur_task: self.prev_task_dist.append([None, None])
            if hasattr(self, 'alignment_model') and self.alignment_model is not None:
                try:
                    if hasattr(self.alignment_model, 'current_mins'):
                        self.prev_task_dist[self._cur_task] = [self.alignment_model.current_mins, self.alignment_model.current_maxs]
                except: pass

        if self.alignment_model and self.adaptive_beta and self.beta_strategy == "performance_based":
            overall_accuracy, _ = self._compute_accuracy(self._network, self.test_loader, self._cur_task)
            self.alignment_model.update_performance(overall_accuracy / 100.0)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        
        if self.replace_augment:
            distribution_vector = np.zeros(self._total_classes, dtype='float')
            distribution_vector, class_counts = log_class_distribution(train_loader, distribution_vector)
            if self._cur_task == 0:
                self.alignment_model = ClipFeatureAlignment(self._cur_task, class_counts, None, None, beta=self.new_beta, device=self._device, adaptive_beta=self.adaptive_beta, beta_strategy=self.beta_strategy, prev_class_counts=self.prev_full_class_counts).to(self._device)
            else:
                prev_min = self.prev_task_dist[self._cur_task-1][0] if len(self.prev_task_dist) > self._cur_task-1 else None
                prev_max = self.prev_task_dist[self._cur_task-1][1] if len(self.prev_task_dist) > self._cur_task-1 else None
                self.alignment_model = ClipFeatureAlignment(self._cur_task, class_counts, prev_min, prev_max, beta=self.new_beta, device=self._device, adaptive_beta=self.adaptive_beta, beta_strategy=self.beta_strategy, prev_class_counts=self.prev_full_class_counts).to(self._device)

        if self.moni_adam and self._cur_task > self.adapter_num - 1: return
        
        epochs = self.args['init_epochs'] if (self._cur_task == 0 or self.init_cls == self.inc) else self.args['later_epochs']
        prog_bar = tqdm(range(epochs))
            
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            
            beta_kl = min(1.0, (epoch + 1) / 10.0) * 0.01

            losses = 0.0
            ce_losses = 0.0
            kl_losses = 0.0
            weighted_kl_losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                if hasattr(self._network.backbone, 'reset_kl'):
                    self._network.backbone.reset_kl()

                if self.replace_augment:
                    inputs, targets_new, mins, maxs = self.alignment_model(inputs, targets, self._cur_task)
                    while len(self.prev_task_dist) <= self._cur_task: self.prev_task_dist.append([None, None])
                    if self._cur_task == 0 or self.prev_task_dist[self._cur_task] != [mins, maxs]: self.prev_task_dist[self._cur_task] = [mins, maxs]
                    if isinstance(targets_new, torch.Tensor): targets = targets_new
                    elif len(targets_new) > 0: targets = torch.stack(targets_new) if isinstance(targets_new[0], torch.Tensor) else torch.tensor(targets_new, dtype=torch.long)
                    else: targets = torch.tensor(targets_new, dtype=torch.long)
                        
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes >= 0, aux_targets - self._known_classes, -1)
                
                output = self._network(inputs, test=False)
                logits = output["logits"]

                ce_loss = F.cross_entropy(logits, aux_targets)
                current_kl = self._network.backbone.get_total_kl() if hasattr(
                    self._network.backbone, 'get_total_kl') else torch.tensor(0.0, device=self._device)
                weighted_kl = beta_kl * current_kl
                loss = ce_loss + weighted_kl

                ce_losses += ce_loss.detach().item()
                kl_losses += current_kl.detach().item()
                weighted_kl_losses += weighted_kl.detach().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler: scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / len(train_loader)
            avg_ce = ce_losses / len(train_loader)
            avg_kl = kl_losses / len(train_loader)
            avg_weighted_kl = weighted_kl_losses / len(train_loader)

            if (epoch + 1) % 5 == 0:
                test_acc, _ = self._compute_accuracy(self._network, test_loader, self._cur_task)
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f} CE {:.3f} KL {:.4f} "
                    "betaKL {:.4f} beta {:.4f}, Train_accy {:.2f}, Test_accy {:.2f}"
                ).format(
                    self._cur_task, epoch + 1, epochs, avg_loss, avg_ce, avg_kl,
                    avg_weighted_kl, beta_kl, train_acc, test_acc)
            else:
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f} CE {:.3f} KL {:.4f} "
                    "betaKL {:.4f} beta {:.4f}, Train_accy {:.2f}"
                ).format(
                    self._cur_task, epoch + 1, epochs, avg_loss, avg_ce, avg_kl,
                    avg_weighted_kl, beta_kl, train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader, cur_task=None):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2), None

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)