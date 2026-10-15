import os
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import MOSNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, target2onehot
from torch.distributions.multivariate_normal import MultivariateNormal

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = MOSNet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.ensemble = args["ensemble"]
        
        # SVP-specific parameters
        self.use_svp = args.get("use_svp", True)
        self.beta_kl = args.get("beta_kl", 0.1)  
        self.kl_warmup_epochs = args.get("kl_warmup_epochs", 5)  

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))
    
    def replace_fc(self):       
        model = self._network.to(self._device)
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader_for_protonet):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.forward_orig(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def after_task(self):
        self._known_classes = self._total_classes
        
        
        if self.use_svp and hasattr(self._network.backbone, 'adapter_update'):
            self._network.backbone.adapter_update()
            logging.info("Updated SVP adapter priors after task completion")

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task
        
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.replace_fc()
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)

        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        
        if hasattr(self._network.backbone, 'adapter_merge'):
            self._network.backbone.adapter_merge()

        self._compute_mean(self._network.backbone)
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)

    def get_optimizer(self, model):
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_fc_params = [p for name, p in model.named_parameters() if 'adapter' not in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.init_lr * 0.1, 'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(network_params, momentum=0.9)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(network_params)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(network_params)

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], 
                gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def evaluate_grouping_granularity(self, train_loader, num_batches=3):
        
        logging.info("\n" + "="*60)
        logging.info("Measuring Gradient Coherence for Grouping Granularity...")
        logging.info("="*60)
        
        self._network.train()
        self._network.zero_grad()
        
        
        for i, (_, inputs, targets) in enumerate(train_loader):
            if i >= num_batches:
                break
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            output = self._network(inputs, adapter_id=self._cur_task, train=True)
            logits = output["logits"][:, :self._total_classes]
            loss = F.cross_entropy(logits, targets.long())
            loss.backward()
            
        
        for layer_idx, adapter in enumerate(self._network.backbone.cur_adapter):
            if hasattr(adapter, 'down_proj') and hasattr(adapter.down_proj, 'q_m'):
                grad = adapter.down_proj.q_m.grad
                if grad is not None:
                    
                    normalized_grad = F.normalize(grad, p=2, dim=1)
                    sim_matrix = torch.mm(normalized_grad, normalized_grad.t())
                    
                    
                    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=self._device)
                    avg_coherence = sim_matrix[mask].mean().item()
                    
                    
                    recommendation = "Block-wise (Coarse)" if avg_coherence > 0.05 else "Channel-wise (Fine)"
                    
                    logging.info(f"Layer {layer_idx:>2} | Avg Gradient Coherence: {avg_coherence:>7.4f} | Optimal: {recommendation}")
                    
        logging.info("="*60 + "\n")
        self._network.zero_grad() 


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        
        
        if self.use_svp and self._cur_task == 0:
            self.evaluate_grouping_granularity(train_loader)
            
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        
        
        epoch_deltas = {}

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            
            
            if self.use_svp:
                if epoch < self.kl_warmup_epochs:
                    current_beta_kl = (epoch / self.kl_warmup_epochs) * self.beta_kl
                else:
                    current_beta_kl = self.beta_kl
            else:
                current_beta_kl = 0.0

            losses = 0.0
            kl_losses = 0.0
            correct, total = 0, 0
            
            
            sparsity_stats = []
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                # Classification loss
                loss_cls = F.cross_entropy(logits, targets.long())
                
                # Orthogonality loss
                loss_orth = self.orth_loss(output['pre_logits'], targets)
                
                
                loss_kl = 0.0
                if self.use_svp and hasattr(self._network.backbone, 'get_kl_loss'):
                    loss_kl = self._network.backbone.get_kl_loss()
                    loss_kl = loss_kl / inputs.size(0)  # Normalize by batch size
                
                # Total loss
                loss = loss_cls + loss_orth + (current_beta_kl * loss_kl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                if self.args["adapter_momentum"] > 0:
                    self._network.backbone.adapter_merge()
                
                losses += loss.item()
                if self.use_svp and loss_kl > 0:
                    kl_losses += loss_kl.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            
            if self.use_svp and hasattr(self._network.backbone, 'get_sparsity_stats'):
                sparsity_stats = self._network.backbone.get_sparsity_stats()

           
            if self.use_svp and hasattr(self._network.backbone, 'get_bias_deltas'):
                epoch_deltas[epoch + 1] = self._network.backbone.get_bias_deltas()

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # Logging
            if self.use_svp and len(sparsity_stats) > 0:
                avg_sparsity = np.mean(sparsity_stats)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, KL {:.3f}, Sparsity {:.2%}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    kl_losses / len(train_loader) if kl_losses > 0 else 0,
                    avg_sparsity,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
        
        
        if self.use_svp and len(sparsity_stats) > 0:
            logging.info(f"Final sparsity per SVP layer: {[f'{s:.2%}' for s in sparsity_stats]}")
            
        
        if self.use_svp and len(epoch_deltas) > 0:
            self.plot_bias_distribution(epoch_deltas, self._cur_task)
            
    def plot_bias_distribution(self, epoch_deltas, task_id):
        
        total_epochs = self.args['tuned_epoch']
        epochs_to_plot = [1, max(2, total_epochs // 2), total_epochs]
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Empirical Distribution of Surrogate Bias (Delta) - Task {task_id}")
        logging.info(f"{'='*60}")
        
        for ep in epochs_to_plot:
            if ep in epoch_deltas:
                deltas = epoch_deltas[ep]
                deltas = deltas[(deltas > -5) & (deltas < 5)]
                
                if len(deltas) == 0:
                    continue
                
                mean_val = np.mean(deltas)
                std_val = np.std(deltas)
                min_val = np.min(deltas)
                max_val = np.max(deltas)
                
                logging.info(f"Epoch {ep} Statistics:")
                logging.info(f"  Mean: {mean_val:>7.4f} | Std: {std_val:>7.4f} | Min: {min_val:>7.4f} | Max: {max_val:>7.4f}")
                
                counts, bin_edges = np.histogram(deltas, bins=10)
                max_count = max(counts) if len(counts) > 0 else 1
                
                logging.info("  Distribution Histogram:")
                for i in range(len(counts)):
                    bin_start = bin_edges[i]
                    bin_end = bin_edges[i+1]
                    bar_len = int(40 * counts[i] / max_count)
                    bar = '#' * bar_len
                    logging.info(f"    [{bin_start:>6.2f}, {bin_end:>6.2f}) | {bar} ({counts[i]})")
                logging.info("-" * 60)
                
        logging.info(f"{'='*60}\n")
        
    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size*3, shuffle=False, num_workers=4
            )
            
            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                
                
                if features_per_cls.shape[0] > 1:
                    raw_cov = torch.cov(features_per_cls.T)
                else:
                    raw_cov = torch.zeros((features_per_cls.shape[-1], features_per_cls.shape[-1]), device=self._device)
                
                
                stable_cov = raw_cov + (torch.eye(features_per_cls.shape[-1], device=self._device) * 1e-4)

                if self.args["ca_storage_efficient_method"] == 'covariance':
                    self.cls_cov[class_idx] = stable_cov
                elif self.args["ca_storage_efficient_method"] == 'variance':
                    self.cls_cov[class_idx] = torch.diag(stable_cov)
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = self.args["n_centroids"]
                features_per_cls = vectors.cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
                kmeans.fit(features_per_cls)
                cluster_lables = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_lables == i]
                    cluster_mean = torch.tensor(
                        np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_var = torch.tensor(
                        np.var(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)
                
                self.cls_mean[class_idx] = cluster_means
                self.cls_cov[class_idx] = cluster_vars

    def classifer_align(self, model):
        model.train()
        
        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device).float()
                    cov = self.cls_cov[class_idx].to(self._device).float()
                    
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)

                    
                    mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
                    cov = torch.nan_to_num(cov, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                    
                    cov = (cov + cov.T) / 2.0 
                    
                    
                    jitter = 1e-4
                    m = None
                    for _ in range(5):
                        try:
                            stable_cov = cov + torch.eye(cov.size(0), device=cov.device) * jitter
                            m = MultivariateNormal(mean, stable_cov)
                            break # Success!
                        except ValueError:
                            jitter *= 10 # Increase jitter and try again
                            
                    o
                    if m is None:
                        logging.warning(f"Covariance for class {class_idx} failed PD check. Using spherical fallback.")
                        fallback_cov = torch.eye(cov.size(0), device=cov.device) * 1e-4
                        m = MultivariateNormal(mean, fallback_cov)

                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([class_idx] * num_sampled_pcls)
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(
                            mean.float(), 
                            (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                logging.info(f"Sampled data shape: {sampled_data.shape}")

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]

                loss = F.cross_entropy(logits, tgt)
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)
         
        logging.info(info)

    def orth_loss(self, features, targets):
        if self.cls_mean:
            sample_mean = []
            for k, v in self.cls_mean.items():
                if isinstance(v, list):
                    sample_mean.extend(v)
                else:
                    sample_mean.append(v)
            sample_mean = torch.stack(sample_mean, dim=0).to(self._device, non_blocking=True)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(
                sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            return self.args["reg"] * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(
                sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            return self.args["reg"] * loss
    
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        orig_y_pred = []
        
        num_saved = len(self._network.backbone.adapter_list)
        available_adapters = num_saved + 1
        
        loop_limit = min(self._cur_task + 1, available_adapters)

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                orig_logits = self._network.forward_orig(inputs)["logits"][:, :self._total_classes]
                orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
                orig_idx = torch.tensor([self.cls2task.get(v, self._cur_task) for v in orig_preds], device=self._device)
                orig_y_pred.append(orig_preds)
                
                all_features = torch.zeros(len(inputs), loop_limit, self._network.backbone.out_dim, device=self._device)
                
                for t_id in range(loop_limit):
                    t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
                    all_features[:, t_id, :] = t_features
                
                final_logits = []
                MAX_ITER = 4
                for x_id in range(len(inputs)):
                    loop_num = 0
                    
                    prev_adapter_idx = orig_idx[x_id]
                    if prev_adapter_idx >= loop_limit:
                        prev_adapter_idx = torch.tensor(loop_limit - 1, device=self._device)

                    while True:
                        loop_num += 1
                        cur_feature = all_features[x_id, prev_adapter_idx].unsqueeze(0)
                        cur_logits = self._network.backbone(cur_feature, fc_only=True)["logits"][:, :self._total_classes]
                        cur_pred = torch.max(cur_logits, dim=1)[1].cpu().numpy()
                        
                        val = cur_pred.item() if hasattr(cur_pred, 'item') else cur_pred
                        cur_adapter_idx = torch.tensor(self.cls2task.get(val, self._cur_task), device=self._device)
                        
                        if cur_adapter_idx >= loop_limit:
                            cur_adapter_idx = torch.tensor(loop_limit - 1, device=self._device)
                        
                        if loop_num >= MAX_ITER or cur_adapter_idx == prev_adapter_idx:
                            break
                        else:
                            prev_adapter_idx = cur_adapter_idx
                    final_logits.append(cur_logits)
                
                final_logits = torch.cat(final_logits, dim=0).to(self._device)

                if self.ensemble:
                    final_logits = F.softmax(final_logits, dim=1)
                    orig_logits = F.softmax(orig_logits / (1/(self._cur_task+1)), dim=1)
                    outputs = final_logits + orig_logits
                else:
                    outputs = final_logits
                
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
        logging.info("the accuracy of the original model:{}".format(np.around(orig_acc, 2)))
        return np.concatenate(y_pred), np.concatenate(y_true)