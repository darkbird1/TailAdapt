import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import SLCANet
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from collections import Counter

# Import ClipFeatureAlignment and related utilities
from utils.panda import log_class_distribution, update_distribution, BalancedSoftmaxLoss
from utils.panda import ClipFeatureAlignment

num_workers = 8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseVariationalLinear(nn.Module):
    
    def __init__(self, in_features, out_features, sparse_init=True,
                 shared_q_a=None, shared_q_b=None,
                 prior_mu=0.01, prior_lambda=5.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.q_m = nn.Parameter(torch.zeros(out_features, in_features))
        self.q_log_s = nn.Parameter(torch.zeros(out_features, in_features))

        
        init_var = (prior_mu ** 3) / max(prior_lambda, 1e-8)
        init_b_sq = math.log1p(init_var / max(prior_mu ** 2, 1e-12))
        init_b = math.sqrt(max(init_b_sq, 1e-12))
        init_a = math.log(prior_mu) - 0.5 * init_b_sq

        if shared_q_a is None:
            self.q_a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        else:
            self.q_a = shared_q_a

        if shared_q_b is None:
            self.q_b = nn.Parameter(
                torch.tensor(math.log(init_b + 1e-12), dtype=torch.float32)
            )
        else:
            self.q_b = shared_q_b

        
        nn.init.zeros_(self.q_m)
        nn.init.zeros_(self.q_log_s)

        self.bias = nn.Parameter(torch.zeros(out_features))

        self.register_buffer("prior_mu", torch.tensor(float(prior_mu)))
        self.register_buffer("prior_lambda", torch.tensor(float(prior_lambda)))
        self.register_buffer(
            "prior_mu0", torch.zeros(out_features, in_features)
        )

        self.kl_loss = torch.tensor(0.0)

    def reset_kl(self):
        self.kl_loss = torch.tensor(
            0.0, device=self.q_m.device, dtype=self.q_m.dtype
        )

    def _moments(self):
        b_sq = torch.exp(2.0 * self.q_b).clamp(min=1e-10, max=20.0)
        mean_z = torch.exp(
            torch.clamp(self.q_a + 0.5 * b_sq, min=-30.0, max=20.0)
        )
        var_z = (
            torch.expm1(b_sq)
            * torch.exp(torch.clamp(2.0 * self.q_a + b_sq,
                                     min=-30.0, max=20.0))
        )
        e_inv_z = torch.exp(
            torch.clamp(-self.q_a + 0.5 * b_sq, min=-30.0, max=30.0)
        )
        return b_sq, mean_z, var_z, e_inv_z

    def conditional_kl(self):
       
        _, _, _, e_inv_z = self._moments()

        s_sq = torch.exp(
            torch.clamp(2.0 * self.q_log_s, min=-30.0, max=20.0)
        )

        mean_term = 0.5 * e_inv_z * (
            self.q_m - self.prior_mu0
        ).square()

        covariance_term = 0.5 * (
            s_sq - 1.0 - 2.0 * self.q_log_s
        )

        return (mean_term + covariance_term).mean()

    def latent_kl(self):
        
        b_sq, mean_z, _, e_inv_z = self._moments()

        mu = self.prior_mu.clamp(min=1e-8)
        lam = self.prior_lambda.clamp(min=1e-8)

        kl = (
            0.5 * self.q_a
            - 0.5 * torch.log(b_sq.clamp(min=1e-10))
            - 0.5
            - 0.5 * torch.log(lam)
            + 0.5 * lam / (mu.square())
            * (mean_z + mu.square() * e_inv_z - 2.0 * mu)
        )

        
        return torch.clamp(kl, min=0.0)

    def compute_kl(self):
        return self.conditional_kl() + self.latent_kl()

    def forward(self, x, train=True):
        if train:
            eps_z = torch.randn((), device=self.q_m.device, dtype=self.q_m.dtype)
            b = torch.exp(self.q_b).clamp(max=5.0)
            z = torch.exp(
                torch.clamp(self.q_a + b * eps_z, min=-30.0, max=20.0)
            )

            eps_w = torch.randn_like(self.q_m)
            sigma = torch.exp(
                torch.clamp(self.q_log_s, min=-10.0, max=5.0)
            )
            weight = self.q_m + torch.sqrt(z) * sigma * eps_w

            self.kl_loss = self.compute_kl()
        else:
            
            weight = self.q_m
            self.kl_loss = torch.zeros(
                (), device=self.q_m.device, dtype=self.q_m.dtype
            )

        return F.linear(x, weight, self.bias)


class SVPAdapter(nn.Module):
    
    def __init__(self, d_model, bottleneck=16):
        super().__init__()

        init_mu = 0.01
        init_lambda = 5.0
        init_var = (init_mu ** 3) / init_lambda
        init_b_sq = math.log1p(init_var / (init_mu ** 2))
        init_b = math.sqrt(max(init_b_sq, 1e-12))
        init_a = math.log(init_mu) - 0.5 * init_b_sq

        self.group_q_a = nn.Parameter(
            torch.tensor(init_a, dtype=torch.float32)
        )
        self.group_q_b = nn.Parameter(
            torch.tensor(math.log(init_b + 1e-12), dtype=torch.float32)
        )

        self.down = SparseVariationalLinear(
            d_model, bottleneck,
            shared_q_a=self.group_q_a,
            shared_q_b=self.group_q_b,
            prior_mu=init_mu,
            prior_lambda=init_lambda,
        )
        self.up = SparseVariationalLinear(
            bottleneck, d_model,
            shared_q_a=self.group_q_a,
            shared_q_b=self.group_q_b,
            prior_mu=init_mu,
            prior_lambda=init_lambda,
        )

        self.act = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.kl_loss = torch.tensor(0.0)

    def reset_kl(self):
        self.down.reset_kl()
        self.up.reset_kl()
        self.kl_loss = torch.zeros(
            (), device=self.scale.device, dtype=self.scale.dtype
        )

    def get_kl(self):
        
        self.kl_loss = (
            self.down.conditional_kl()
            + self.up.conditional_kl()
            + self.down.latent_kl()
        )
        return self.kl_loss

    def forward(self, x, train=True):
        resid = x
        x = self.down(x, train=train)
        x = self.act(x)
        x = self.up(x, train=train)
        return resid + x * self.scale

    def update_prior_from_posterior(self, rho=0.05):
        
        with torch.no_grad():
            _, mean_z, var_z, _ = self.down._moments()
            var_z = var_z.clamp(min=1e-10)

            hat_mu = mean_z
            hat_lambda = hat_mu.pow(3) / var_z

            # Same group prior for both projections.
            old_mu = self.down.prior_mu
            old_lambda = self.down.prior_lambda

            new_mu = (1.0 - rho) * old_mu + rho * hat_mu
            new_lambda = (1.0 - rho) * old_lambda + rho * hat_lambda

            old_mu0_down = self.down.prior_mu0
            old_mu0_up = self.up.prior_mu0

            new_mu0_down = (
                (1.0 - rho) * old_mu0_down + rho * self.down.q_m
            )
            new_mu0_up = (
                (1.0 - rho) * old_mu0_up + rho * self.up.q_m
            )

            self.down.prior_mu.copy_(new_mu)
            self.up.prior_mu.copy_(new_mu)
            self.down.prior_lambda.copy_(new_lambda)
            self.up.prior_lambda.copy_(new_lambda)
            self.down.prior_mu0.copy_(new_mu0_down)
            self.up.prior_mu0.copy_(new_mu0_up)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SLCANet(args, pretrained=True)

        self.svp_adapters = nn.ModuleList()

        
        if hasattr(self._network, 'backbone') and hasattr(
            self._network.backbone, 'blocks'
        ):
            for i, block in enumerate(self._network.backbone.blocks):
                
                d_model = getattr(
                    self._network.backbone,
                    "embed_dim",
                    getattr(self._network.backbone, "num_features", 768)
                )

                adapter = SVPAdapter(
                    d_model=d_model,
                    bottleneck=args.get("svp_bottleneck", 16)
                ).to(self._device)

                self.svp_adapters.append(adapter)

                original_forward = block.forward

                def new_forward(
                    x, original_f=original_forward, adapt=adapter
                ):
                    x = original_f(x)
                    x = adapt(x, train=adapt.training)
                    return x

                block.forward = new_forward

        
        for p in self._network.parameters():
            p.requires_grad = False

        for p in self.svp_adapters.parameters():
            p.requires_grad = True

        
        self._classifier = None

        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.lrate_decay = args['lrate_decay']
        self.weight_decay = args['weight_decay']
        self.milestones = args['milestones']

        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0 / 100

        
        self.fix_bcb = (self.bcb_lrscale == 0)

        self.ca_epochs = args['ca_epochs']

        if (
            'ca_with_logit_norm' in args.keys()
            and args['ca_with_logit_norm'] > 0
        ):
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None

        self.save_before_ca = bool(
            args.get('save_before_ca', False)
        )

        self.args = args
        self.seed = args['seed']
        self.task_sizes = []

        self.replace_augment = args.get("replace_augment", True)
        self.prev_task_dist = []
        self.beta = args.get("beta", 0.5)
        self.prev_full_class_counts = Counter()

        self.adaptive_beta = args.get("adaptive_beta", True)
        self.beta_strategy = args.get("beta_strategy", "task_progress")

        self.alignment_model = None
        self.task_accuracies = []

        
        self.tailadapt_rho = args.get("tailadapt_rho", 0.05)
        self.beta_kl_max = args.get("beta_kl", 0.01)
        self.beta_kl_warmup = args.get("beta_kl_warmup", 10)
    

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        if self._network.fc is not None and hasattr(self._network.fc, 'recall'):
            self._network.fc.recall()
        
        
        self._update_for_next_task()
        self._update_svp_priors()
    
        self._known_classes = self._total_classes
    def _update_svp_priors(self):
        
        for adapt in self.svp_adapters:
            adapt.update_prior_from_posterior(rho=self.tailadapt_rho)

    def _update_for_next_task(self):
        
        if hasattr(self, 'train_dataset') and self.train_dataset is not None and len(self.train_dataset) > 0:
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
            logging.info(f"Task {self._cur_task} finished. Storing its distribution for next task: {self.prev_full_class_counts}")
        else:
            logging.info(f"Task {self._cur_task}: No train_dataset or empty dataset found for prev_full_class_counts update.")

        # Update performance history for performance-based beta adaptation
        if self.alignment_model and self.adaptive_beta and self.beta_strategy == "performance_based":
            overall_accuracy = self._compute_accuracy(self._network, self.test_loader, self._cur_task, self._total_classes)
            self.alignment_model.update_performance(overall_accuracy / 100.0)
            logging.info(f"Task {self._cur_task}: Performance for beta adaptation: {overall_accuracy:.2f}%")

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5
        # SLCANet.update_fc() creates/replaces self._network.fc.
        self._network.update_fc(
            data_manager.get_task_size(self._cur_task)
        )

       
        self._classifier = self._network.fc

        if self._classifier is None:
            raise RuntimeError(
                "SLCANet.update_fc() returned with self._network.fc=None."
            )

        logging.info(
            'Learning on {}-{}'.format(
                self._known_classes, self._total_classes
            )
        )

        self._network.to(self._device)

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[])
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()

        # Store datasets for later use
        self.train_dataset = train_dset
        self.data_manager = data_manager

        self.train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self._stage1_training(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        
        self._compute_class_mean(data_manager)
        if self._cur_task > 0 and self.ca_epochs > 0:
            self._stage2_compact_classifier(task_size)

    def _setup_clip_alignment(self, train_loader):
        """Setup ClipFeatureAlignment model"""
        if self.replace_augment:
            distribution_vector = np.zeros(self._total_classes, dtype='float')
            distribution_vector, class_counts = log_class_distribution(train_loader, distribution_vector)
            
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

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        self._setup_clip_alignment(train_loader)

        prog_bar = tqdm(range(self.epochs))

        for epoch in prog_bar:
            if self.beta_kl_warmup <= 0:
                beta_kl = self.beta_kl_max
            else:
                beta_kl = self.beta_kl_max * min(
                    1.0, float(epoch + 1) / float(self.beta_kl_warmup)
                )

            self._network.train()
            for adapt in self.svp_adapters:
                adapt.train()

            losses = 0.0
            ce_losses = 0.0
            kl_losses = 0.0
            weighted_kl_losses = 0.0
            sparsity_sum = 0.0

            first_batch_processed = False

            for i, (_, inputs, targets) in enumerate(train_loader):
                if (
                    self.replace_augment
                    and self.alignment_model is not None
                ):
                    inputs, targets, mins, maxs = self.alignment_model(
                        inputs, targets, self._cur_task
                    )

                    if not first_batch_processed:
                        current_beta = (
                            self.alignment_model.get_current_beta()
                        )
                        logging.info(
                            f"Task {self._cur_task}: "
                            f"Using adaptive beta = {current_beta:.3f} "
                            f"(base: {self.beta:.3f}, "
                            f"strategy: {self.beta_strategy})"
                        )
                        first_batch_processed = True

                    if (
                        self._cur_task == 0
                        or (
                            len(self.prev_task_dist) >= self._cur_task
                            and self.prev_task_dist[self._cur_task - 1]
                            != [mins, maxs]
                        )
                    ):
                        self.prev_task_dist.append([mins, maxs])

                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(
                            targets, dtype=torch.long, device=inputs.device
                        )

                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

               
                for adapt in self.svp_adapters:
                    adapt.reset_kl()

                logits = self._network(
                    inputs, bcb_no_grad=self.fix_bcb
                )['logits']

                cur_targets = torch.where(
                    targets - self._known_classes >= 0,
                    targets - self._known_classes,
                    -100
                )

                loss_cls = F.cross_entropy(
                    logits[:, self._known_classes:],
                    cur_targets
                )

                
                loss_kl = torch.zeros(
                    (), device=self._device, dtype=loss_cls.dtype
                )

                for adapt in self.svp_adapters:
                    loss_kl = loss_kl + adapt.get_kl()

                    with torch.no_grad():
                        z_mean = torch.exp(adapt.group_q_a)
                        sparsity_sum += (
                            z_mean > 0.01
                        ).float()

                
                loss = loss_cls + beta_kl * loss_kl

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.svp_adapters.parameters(), max_norm=5.0
                )
                optimizer.step()

                losses += float(loss.detach())
                ce_losses += float(loss_cls.detach())
                kl_losses += float(loss_kl.detach())
                weighted_kl_losses += float(
                    (beta_kl * loss_kl).detach()
                )

            scheduler.step()

            train_acc = self._compute_accuracy(
                self._network,
                train_loader,
                self._cur_task,
                self._total_classes
            )

            avg_sparsity = (
                sparsity_sum / max(len(self.svp_adapters), 1)
            )

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(
                    self._network,
                    test_loader,
                    self._cur_task,
                    self._total_classes
                )
                info = (
                    "Task {}, Epoch {}/{} => "
                    "Loss {:.4f}, CE {:.4f}, KL {:.4f}, "
                    "betaKL {:.4f}, beta {:.4f}, "
                    "Train_accy {:.2f}, Test_accy {:.2f}, "
                    "Sparsity {:.3f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / max(len(train_loader), 1),
                    ce_losses / max(len(train_loader), 1),
                    kl_losses / max(len(train_loader), 1),
                    weighted_kl_losses / max(len(train_loader), 1),
                    beta_kl,
                    train_acc,
                    test_acc,
                    avg_sparsity
                )
            else:
                info = (
                    "Task {}, Epoch {}/{} => "
                    "Loss {:.4f}, CE {:.4f}, KL {:.4f}, "
                    "betaKL {:.4f}, beta {:.4f}, "
                    "Train_accy {:.2f}, Sparsity {:.3f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / max(len(train_loader), 1),
                    ce_losses / max(len(train_loader), 1),
                    kl_losses / max(len(train_loader), 1),
                    weighted_kl_losses / max(len(train_loader), 1),
                    beta_kl,
                    train_acc,
                    avg_sparsity
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        
        for p in self._network.backbone.parameters():
            p.requires_grad = not self.fix_bcb

        for p in self.svp_adapters.parameters():
            p.requires_grad = True

        
        self._classifier = None

        self._classifier = self._network.fc
        if self._classifier is None:
            raise RuntimeError(
                "self._network.fc is None at Stage-1. "
                "SLCANet.update_fc() must run before _stage1_training()."
            )
        for p in self._classifier.parameters():
            p.requires_grad = True

        adapter_params = [
            p for p in self.svp_adapters.parameters()
            if p.requires_grad
        ]

        fc_params = [
            p for p in self._classifier.parameters()
            if p.requires_grad
        ]

        network_params = [
            {
                'params': adapter_params,
                'lr': self.lrate,
                'weight_decay': self.weight_decay
            },
            {
                'params': fc_params,
                'lr': self.lrate,
                'weight_decay': self.weight_decay
            }
        ]

        if not self.fix_bcb:
            backbone_params = [
                p for p in self._network.backbone.parameters()
                if p.requires_grad
            ]
            network_params.insert(
                0,
                {
                    'params': backbone_params,
                    'lr': self.lrate * self.bcb_lrscale,
                    'weight_decay': self.weight_decay
                }
            )

        optimizer = optim.SGD(
            network_params,
            lr=self.lrate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.milestones,
            gamma=self.lrate_decay
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(
                self._network, self._multiple_gpus
            )

        self._run(
            train_loader,
            test_loader,
            optimizer,
            scheduler
        )

    def _stage2_compact_classifier(self, task_size):
        for p in self._network.parameters():
            p.requires_grad = False
        for p in self._classifier.parameters():
            p.requires_grad = True
            
        run_epochs = self.ca_epochs
        crct_num = self._total_classes    
        self._classifier = self._network.fc
        param_list = [p for p in self._classifier.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.lrate,
                           'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = c_id//task_size
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)*(0.9+decay)
                cls_cov = self._class_covs_slca[c_id].to(self._device)
                
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader, self._cur_task, self._total_classes)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)

    def _compute_class_mean(self, data_manager):
        if hasattr(self, '_class_means_slca') and self._class_means_slca is not None:
            ori_classes = self._class_means_slca.shape[0]
            assert ori_classes==self._known_classes
            new_class_means_slca = np.zeros((self._total_classes, self.feature_dim))
            new_class_means_slca[:self._known_classes] = self._class_means_slca
            self._class_means_slca = new_class_means_slca
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs_slca
            self._class_covs_slca = new_class_cov
        else:
            self._class_means_slca = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs_slca = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = np.mean(vectors, axis=0)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
            self._class_means_slca[class_idx, :] = class_mean
            self._class_covs_slca[class_idx, ...] = class_cov