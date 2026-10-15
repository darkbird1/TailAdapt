import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

from utils.panda import ClipFeatureAlignment, log_class_distribution, update_distribution, BalancedSoftmaxLoss

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["backbone_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')

        if 'resnet' in args['backbone_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self.batch_size=128
            self.init_lr=args["init_lr"] if args["init_lr"] is not None else 0.01
        else:
            self._network = SimpleVitNet(args, True)
            self.batch_size = args["batch_size"]
            self.init_lr = args["init_lr"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        self.replace_augment = args["replace_augment"]
        self.prev_task_dist = []
        
        

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        if self.replace_augment:
            distribution_vector = np.zeros(self._total_classes, dtype='float')
            distribution_vector, class_counts = log_class_distribution(trainloader, distribution_vector)
            #clip model for alignment - dist vec, device
            if self._cur_task == 0:
                alignment_model = ClipFeatureAlignment(class_counts, prev_min=0, prev_max=0, beta=0.5, device=self._device)
            else:
                prev_min = self.prev_task_dist[self._cur_task-1][0]
                prev_max = self.prev_task_dist[self._cur_task-1][1]
                alignment_model = ClipFeatureAlignment(class_counts, prev_min, prev_max, beta=0.5, device=self._device)

        embedding_list = []
        label_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                if self.replace_augment:
                    data, labels, mins, maxs = alignment_model(data, label, self._cur_task)
                    if self.prev_task_dist[self._cur_task - 1] != [mins, maxs]:
                        self.prev_task_dist.append([mins, maxs])
                        
                data=data.to(self._device)
                label=label.to(self._device)
                # Backbone in eval mode -> Sparse features
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.fc.weight.data[class_index]=proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)

        if self._cur_task == 0:
            total_params = sum(
                p.numel() for p in self._network.parameters()
            )
            print(f"{total_params:,} total parameters.")

            
            latent_params = []
            adapter_means = []
            std_params = []

            for name, p in self._network.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in name for k in ["q_a", "q_b"]):
                    latent_params.append(p)
                elif "q_m" in name:
                    adapter_means.append(p)
                else:
                    std_params.append(p)

            param_groups = [
                {
                    "params": latent_params,
                    "lr": self.init_lr,
                    "weight_decay": 0.0,
                },
                {
                    "params": adapter_means,
                    "lr": self.init_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": std_params,
                    "lr": self.init_lr,
                    "weight_decay": self.weight_decay,
                },
            ]

            if self.args["optimizer"] == "sgd":
                optimizer = optim.SGD(
                    param_groups,
                    momentum=0.9
                )
            elif self.args["optimizer"] == "adam":
                optimizer = optim.AdamW(param_groups)
            else:
                raise ValueError(
                    f"Unsupported optimizer: {self.args['optimizer']}"
                )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args["tuned_epoch"],
                eta_min=self.min_lr,
            )

            self._init_train(
                train_loader,
                test_loader,
                optimizer,
                scheduler
            )

            
            if hasattr(self._network.backbone, "update_svp_priors"):
                logging.info(
                    "TailAdapt: updating priors at task boundary..."
                )
                self._network.backbone.update_svp_priors(rho=0.05)

            self.construct_dual_branch_network()

        else:
            pass

        self.replace_fc(
            train_loader_for_protonet,
            self._network,
            None
        )

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["tuned_epoch"]))

        if self.replace_augment:
            distribution_vector = np.zeros(
                self._total_classes,
                dtype="float"
            )
            distribution_vector, class_counts = log_class_distribution(
                train_loader,
                distribution_vector
            )

            if self._cur_task == 0:
                alignment_model = ClipFeatureAlignment(
                    class_counts,
                    prev_min=0,
                    prev_max=0,
                    beta=0.5,
                    device=self._device
                )
            else:
                prev_min = self.prev_task_dist[self._cur_task - 1][0]
                prev_max = self.prev_task_dist[self._cur_task - 1][1]
                alignment_model = ClipFeatureAlignment(
                    class_counts,
                    prev_min,
                    prev_max,
                    beta=0.5,
                    device=self._device
                )
            alignment_model.to(self._device)

        # KL coefficient is deliberately small and smoothly introduced.
        target_beta = float(
            self.args.get("tailadapt_beta", 1e-5)
        )
        warmup_epochs = min(
            5,
            max(0, self.args["tuned_epoch"] - 1)
        )

        for epoch in prog_bar:
            self._network.train()

            
            if self._cur_task == 0:
                beta_kl = 0.0
            elif epoch < warmup_epochs:
                beta_kl = 0.0
            else:
                denom = max(
                    1,
                    self.args["tuned_epoch"] - warmup_epochs - 1
                )
                progress = (
                    epoch - warmup_epochs
                ) / denom
                beta_kl = target_beta * min(
                    1.0,
                    max(0.0, progress)
                )

            losses = 0.0
            kl_losses = 0.0
            kl_group_losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.replace_augment:
                    inputs, targets, mins, maxs = alignment_model(
                        inputs,
                        targets,
                        self._cur_task
                    )

                    if (
                        self._cur_task == 0
                        or (
                            len(self.prev_task_dist) >= self._cur_task
                            and self.prev_task_dist[
                                self._cur_task - 1
                            ] != [mins, maxs]
                        )
                    ):
                        self.prev_task_dist.append([mins, maxs])

                    targets = torch.stack(targets)

                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                if hasattr(self._network.backbone, "reset_kl"):
                    self._network.backbone.reset_kl()

                logits = self._network(inputs)["logits"]
                ce_loss = F.cross_entropy(logits, targets)

                current_kl = torch.zeros(
                    (),
                    device=self._device,
                    dtype=ce_loss.dtype
                )
                kl_per_group = current_kl

                if hasattr(
                    self._network.backbone,
                    "get_kl_statistics"
                ):
                    stats = self._network.backbone.get_kl_statistics()
                    current_kl = stats["raw_kl"]
                    kl_per_group = stats["kl_per_group"]

                
                loss = ce_loss + beta_kl * kl_per_group

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                losses += loss.item()
                kl_losses += float(current_kl.detach().item())
                kl_group_losses += float(
                    kl_per_group.detach().item()
                )

                _, preds = torch.max(logits, dim=1)
                correct += (
                    preds.eq(targets.expand_as(preds))
                    .cpu()
                    .sum()
                    .item()
                )
                total += len(targets)

            scheduler.step()

            train_acc = np.around(
                correct * 100.0 / max(1, total),
                decimals=2
            )
            avg_kl = kl_losses / max(1, len(train_loader))
            avg_kl_group = (
                kl_group_losses /
                max(1, len(train_loader))
            )

            test_acc = self._compute_accuracy(
                self._network,
                test_loader,
                self._cur_task,
                self._total_classes
            )

            info = (
                "Task {}, Epoch {}/{} => "
                "Loss {:.3f}, KL {:.6f}, KL/group {:.6f}, "
                "beta {:.2e}, Train {:.2f}, Test {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                losses / max(1, len(train_loader)),
                avg_kl,
                avg_kl_group,
                beta_kl,
                train_acc,
                test_acc,
            )

            prog_bar.set_description(info)

        logging.info(info)

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network=network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        
        # [Augmentation setup - unchanged]
        if self.replace_augment:
            distribution_vector = np.zeros(self._total_classes, dtype='float')
            distribution_vector, class_counts = log_class_distribution(train_loader, distribution_vector)
            if self._cur_task == 0:
                alignment_model = ClipFeatureAlignment(class_counts, prev_min=0, prev_max=0, beta=0.5, device=self._device)
                alignment_model.to(self._device)
            else:
                prev_min = self.prev_task_dist[self._cur_task-1][0]
                prev_max = self.prev_task_dist[self._cur_task-1][1]
                alignment_model = ClipFeatureAlignment(class_counts, prev_min, prev_max, beta=0.5, device=self._device)
                alignment_model.to(self._device)

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            
            warmup_epochs = 5
            target_beta = 0.01
            if epoch < warmup_epochs: beta_kl = 0.0
            else: beta_kl = ((epoch - warmup_epochs) / (self.args['tuned_epoch'] - warmup_epochs)) * target_beta

            losses = 0.0
            kl_losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.replace_augment:
                    inputs, targets, mins, maxs = alignment_model(inputs, targets, self._cur_task)
                    if self._cur_task == 0 or (len(self.prev_task_dist) >= self._cur_task and self.prev_task_dist[self._cur_task - 1] != [mins, maxs]):
                        self.prev_task_dist.append([mins, maxs])
                    targets = torch.stack(targets)
                
               
                if hasattr(self._network.backbone, 'reset_kl'):
                    self._network.backbone.reset_kl()

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                
                # [SVP] Add KL
                if hasattr(self._network.backbone, 'get_total_kl'):
                    current_kl = self._network.backbone.get_total_kl()
                    # Scale factor 1e-4 keeps it safe relative to CE loss
                    loss += (beta_kl * current_kl * 1e-4) 
                    if isinstance(current_kl, torch.Tensor):
                        kl_losses += current_kl.item()

                optimizer.zero_grad()
                loss.backward()
                
                
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_kl = kl_losses / len(train_loader)

            test_acc = self._compute_accuracy(self._network, test_loader, self._cur_task, self._total_classes)
            info = "Task {}, Epoch {}/{} => Loss {:.3f} KL {:.2f}, Train {:.2f}, Test {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'],
                losses / len(train_loader), avg_kl, train_acc, test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)