# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
import logging
import copy




class SparseVariationalLinear(nn.Module):
    
    def __init__(self, in_features, out_features, sparse_init=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        
        self.q_m = nn.Parameter(torch.empty(out_features, in_features))
        self.q_log_s = nn.Parameter(
            torch.zeros(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.q_m, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.zeros(out_features))

        
        self.register_buffer("prior_mu", torch.tensor(0.1))
        self.register_buffer("prior_lambda", torch.tensor(5.0))
        self.register_buffer(
            "prior_mu0",
            torch.zeros(out_features, in_features)
        )

        self.kl_loss = torch.zeros(())

    def forward(self, x, z, train=True):
        
        weight = self.q_m

        return F.linear(x, weight, self.bias)

    def conditional_kl(self, e_inv_z):
        
        s_sq = torch.exp(2.0 * self.q_log_s)

        mean_term = e_inv_z * (
            self.q_m - self.prior_mu0
        ).pow(2)

        cov_term = (
            s_sq
            - 1.0
            - torch.log(s_sq.clamp_min(1e-12))
        )

        return 0.5 * (
            mean_term.sum() + cov_term.sum()
        )


class SVPAdapter(nn.Module):
    
    def __init__(
        self,
        d_model,
        bottleneck=16,
        dropout=0.0,
        adapter_scalar="1.0",
        adapter_layernorm_option="in",
    ):
        super().__init__()

        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(
                self.n_embd
            )

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = SparseVariationalLinear(
            self.n_embd,
            self.down_size,
        )
        self.non_linear_func = nn.ReLU()
        self.up_proj = SparseVariationalLinear(
            self.down_size,
            self.n_embd,
        )

        self.dropout = dropout

        
        init_mu = 0.1
        init_lambda = 5.0
        init_var = init_mu ** 3 / init_lambda

        init_b_sq = math.log1p(
            init_var / (init_mu ** 2)
        )
        init_b = math.sqrt(init_b_sq)
        init_a = math.log(init_mu) - 0.5 * init_b_sq

        self.q_a = nn.Parameter(
            torch.tensor(init_a, dtype=torch.float32)
        )
        self.q_b = nn.Parameter(
            torch.tensor(math.log(init_b), dtype=torch.float32)
        )

        # Group-level IG prior.
        self.register_buffer(
            "prior_mu",
            torch.tensor(init_mu, dtype=torch.float32)
        )
        self.register_buffer(
            "prior_lambda",
            torch.tensor(init_lambda, dtype=torch.float32)
        )

        
        self.register_buffer(
            "prior_mu0_down",
            self.down_proj.q_m.detach().clone()
        )
        self.register_buffer(
            "prior_mu0_up",
            self.up_proj.q_m.detach().clone()
        )

        self.kl_loss = torch.zeros(())

    
    def _moments(self):
        b_sq = torch.exp(
            2.0 * self.q_b
        ).clamp(max=20.0)

        e_z = torch.exp(
            self.q_a + 0.5 * b_sq
        )

        var_z = (
            torch.expm1(b_sq)
            * torch.exp(
                2.0 * self.q_a + b_sq
            )
        )

        e_inv_z = torch.exp(
            -self.q_a + 0.5 * b_sq
        )

        return e_z, var_z, e_inv_z

    
    def latent_kl(self):
        
        b_sq = torch.exp(
            2.0 * self.q_b
        ).clamp(max=20.0)

        mu = self.prior_mu.clamp_min(1e-8)
        lam = self.prior_lambda.clamp_min(1e-8)

        e_z, _, e_inv_z = self._moments()

        kl = (
            0.5 * self.q_a
            - 0.5 * torch.log(
                b_sq.clamp_min(1e-12)
            )
            + (
                lam / (2.0 * mu.pow(2))
            )
            * (
                e_z
                + mu.pow(2) * e_inv_z
                - 2.0 * mu
            )
        )

        return kl

   
    def compute_kl(self):
        _, _, e_inv_z = self._moments()

        self.down_proj.prior_mu0 = self.prior_mu0_down
        self.up_proj.prior_mu0 = self.prior_mu0_up

        conditional_kl = (
            self.down_proj.conditional_kl(e_inv_z)
            + self.up_proj.conditional_kl(e_inv_z)
        )

        # z_g is shared by down/up, so latent KL is counted ONCE.
        latent_kl = self.latent_kl()

        return conditional_kl + latent_kl

    
    def forward(
        self,
        x,
        add_residual=True,
        residual=None,
        train=True,
    ):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        
        e_z, _, _ = self._moments()

        down = self.down_proj(
            x,
            z=e_z,
            train=train,
        )
        down = self.non_linear_func(down)

        down = F.dropout(
            down,
            p=self.dropout,
            training=self.training,
        )

        up = self.up_proj(
            down,
            z=e_z,
            train=train,
        )

        if train:
            self.kl_loss = self.compute_kl()
        else:
            self.kl_loss = torch.zeros(
                (),
                device=x.device,
                dtype=x.dtype,
            )

        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            return up + residual

        return up

    
    def update_priors(self, rho=0.05):
        
        with torch.no_grad():
            e_z, var_z, _ = self._moments()

            hat_mu = e_z.clamp_min(1e-6)

            hat_lambda = (
                hat_mu.pow(3)
                / var_z.clamp_min(1e-8)
            ).clamp(1e-6, 1e8)

            self.prior_mu.mul_(1.0 - rho).add_(
                rho * hat_mu
            )

            self.prior_lambda.mul_(1.0 - rho).add_(
                rho * hat_lambda
            )

            self.prior_mu0_down.mul_(1.0 - rho).add_(
                rho * self.down_proj.q_m.detach()
            )

            self.prior_mu0_up.mul_(1.0 - rho).add_(
                rho * self.up_proj.q_m.detach()
            )

    def reset_kl(self):
        self.kl_loss = torch.zeros(
            (),
            device=self.q_a.device,
            dtype=self.q_a.dtype,
        )

    def get_group_scale(self):
        with torch.no_grad():
            e_z, _, _ = self._moments()
            return e_z.detach()

    def get_sparsity(self, threshold=0.01):
        """
        Group activity, NOT channel-wise/parameter-wise sparsity.
        One adapter produces exactly one binary activity value.
        """
        with torch.no_grad():
            e_z, _, _ = self._moments()
            return (
                (e_z > threshold)
                .float()
            )




class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    def reset_kl(self):
        return




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None, train=False):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        if adapt is not None:
            # Handle both SVP and regular adapters
            if isinstance(adapt, SVPAdapter):
                adapt_x = adapt(x, add_residual=False, train=train)
            else:
                adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    if isinstance(adapt, SVPAdapter):
                        x = adapt(x, train=train)
                    else:
                        x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt)

        x = residual + x

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for mixed SVP and regular adapters"""
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, 
                 distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, 
                 norm_layer=None, act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        print("I'm using ViT with mixed SVP and regular adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            self.embeddings = nn.ParameterList(
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)
         
        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = nn.ModuleList()
        self.cur_adapter = nn.ModuleList()
        
        # Storage for adapter momentum
        self.down_weight_sum = [[] for i in range(len(self.blocks))]
        self.down_bias_sum = [[] for i in range(len(self.blocks))]
        self.up_weight_sum = [[] for i in range(len(self.blocks))]
        self.up_bias_sum = [[] for i in range(len(self.blocks))]
        
        # SVP-specific tracking
        self.svp_fraction = getattr(tuning_config, 'svp_fraction', 0.5)  # Fraction of SVP adapters
        self.is_svp_layer = []  # Track which layers use SVP
        
        self.init_adapters()

    def init_adapters(self):
        """Initialize mixed SVP and regular adapters"""
        config = self.config
        self.cur_adapter = nn.ModuleList()
        self.is_svp_layer = []
        
        num_layers = len(self.blocks)
        num_svp = int(num_layers * self.svp_fraction)
        
        
        svp_indices = set(torch.linspace(0, num_layers-1, num_svp).long().tolist())
        num_layers = len(self.blocks)
        num_svp = int(num_layers * self.svp_fraction)
        svp_start_idx = num_layers - num_svp
        
        for i in range(num_layers):
            if i >= svp_start_idx:
                
                adapter = SVPAdapter(
                    d_model=config.d_model,
                    bottleneck=config.ffn_num,
                    dropout=0.1,
                    adapter_scalar=config.ffn_adapter_scalar,
                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                ).to(self._device)
                self.is_svp_layer.append(True)
                logging.info(f"Layer {i}: Using SVP Adapter")
            else:
                
                adapter = Adapter(
                    self.config, 
                    dropout=0.1, 
                    bottleneck=config.ffn_num,
                    init_option=config.ffn_adapter_init_option,
                    adapter_scalar=config.ffn_adapter_scalar,
                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                ).to(self._device)
                self.is_svp_layer.append(False)
                logging.info(f"Layer {i}: Using Regular Adapter")
        
        
            
            self.cur_adapter.append(adapter)
        
        self.cur_adapter.requires_grad_(True)
        logging.info(f"Initialized {num_svp} SVP adapters and {num_layers - num_svp} regular adapters")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.blocks)):
            self.cur_adapter[i].requires_grad = True

    def adapter_merge(self):
        
        adapter_momentum = self.config.adapter_momentum
        if len(self.adapter_list) == 0 or adapter_momentum == 0:
            pass
        else:   
            self.cur_adapter = self.reweight_adapter(self.cur_adapter, len(self.adapter_list))
    
    def adapter_update(self):
        self.reset_kl()
        
        self.adapter_list.append(copy.deepcopy(self.cur_adapter))
        self.sum_adapter_param()
        
        
        for i, adapter in enumerate(self.cur_adapter):
            if self.is_svp_layer[i]:
                adapter.update_priors(rho=getattr(self.config, 'tailadapt_rho', 0.05))
    
    def sum_adapter_param(self):
        
        for layer_idx in range(len(self.blocks)):
            if self.is_svp_layer[layer_idx]:
                
                continue
            
            adapter = self.cur_adapter[layer_idx]
            if len(self.down_weight_sum[layer_idx]) > 0:
                self.down_weight_sum[layer_idx].append(
                    adapter.down_proj.weight.data + self.down_weight_sum[layer_idx][-1])
                self.down_bias_sum[layer_idx].append(
                    adapter.down_proj.bias.data + self.down_bias_sum[layer_idx][-1])
                self.up_weight_sum[layer_idx].append(
                    adapter.up_proj.weight.data + self.up_weight_sum[layer_idx][-1])
                self.up_bias_sum[layer_idx].append(
                    adapter.up_proj.bias.data + self.up_bias_sum[layer_idx][-1])
            else:
                self.down_weight_sum[layer_idx].append(adapter.down_proj.weight.data)
                self.down_bias_sum[layer_idx].append(adapter.down_proj.bias.data)
                self.up_weight_sum[layer_idx].append(adapter.up_proj.weight.data)
                self.up_bias_sum[layer_idx].append(adapter.up_proj.bias.data)
    
    def reweight_adapter(self, adapter, idx):
        "
        momentum = self.config.adapter_momentum
        if momentum == 0 or idx == 0:
            return adapter

        for layer_idx in range(len(self.blocks)):
            if self.is_svp_layer[layer_idx]:
                # SVP adapters don't use momentum
                continue
            
            adapter[layer_idx].down_proj.weight.data = (1 - momentum) * adapter[layer_idx].down_proj.weight.data \
                + momentum * self.down_weight_sum[layer_idx][idx-1] / idx
            adapter[layer_idx].down_proj.bias.data = (1 - momentum) * adapter[layer_idx].down_proj.bias.data \
                + momentum * self.down_bias_sum[layer_idx][idx-1] / idx
            adapter[layer_idx].up_proj.weight.data = (1 - momentum) * adapter[layer_idx].up_proj.weight.data \
                + momentum * self.up_weight_sum[layer_idx][idx-1] / idx
            adapter[layer_idx].up_proj.bias.data = (1 - momentum) * adapter[layer_idx].up_proj.bias.data \
                + momentum * self.up_bias_sum[layer_idx][idx-1] / idx
        
        return adapter
    def reset_kl(self):
        
        for adapt in self.cur_adapter:
            adapt.reset_kl()
    def get_kl_loss(self, normalize=True):
        
        kl_loss = None
        n_params = 0

        for i, adapter in enumerate(self.cur_adapter):
            if self.is_svp_layer[i] and hasattr(adapter, 'kl_loss'):
                value = adapter.kl_loss
                kl_loss = (
                    value if kl_loss is None
                    else kl_loss + value
                )

                n_params += (
                    adapter.down_proj.q_m.numel()
                    + adapter.down_proj.q_log_s.numel()
                    + adapter.up_proj.q_m.numel()
                    + adapter.up_proj.q_log_s.numel()
                    + 2
                )

        if kl_loss is None:
            return torch.zeros(
                (),
                device=self.cls_token.device,
                dtype=self.cls_token.dtype,
            )

        if normalize:
            return kl_loss / max(n_params, 1)

        return kl_loss
    
    def get_sparsity_stats(self):
        
        activity = []
        for i, adapter in enumerate(self.cur_adapter):
            if self.is_svp_layer[i]:
                activity.append(adapter.get_sparsity().item())
        return activity
    
    def forward_features(self, x, adapter_id, train):       
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        if adapter_id == -1:
            x = self.blocks(x)
        else:
            for layer_idx, blk in enumerate(self.blocks):
                if train:
                    orig_adapter = self.cur_adapter
                    momentum_adapter = self.reweight_adapter(orig_adapter, adapter_id)
                    x = blk(x, momentum_adapter[layer_idx], train=train)
                else:
                    #print(adapter_id,len(self.adapter_list))
                    if adapter_id == len(self.adapter_list):
                        orig_adapter = self.cur_adapter
                        momentum_adapter = self.reweight_adapter(orig_adapter, adapter_id)
                        x = blk(x, momentum_adapter[layer_idx], train=train)
                    elif adapter_id < len(self.adapter_list):
                        orig_adapter = self.adapter_list[adapter_id]
                        momentum_adapter = self.reweight_adapter(orig_adapter, adapter_id)
                        x = blk(x, momentum_adapter[layer_idx], train=train)
                    else:
                        raise ValueError("adapter_id is wrong.")

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        res = dict()
        res['x'] = outcome
        res['pre_logits'] = outcome
        res['features'] = outcome
        
        return res

    def forward_head(self, res):
        x = res['x']
        res['logits'] = self.head(x)
        return res

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        if fc_only:
            res = dict()
            res['logits'] = self.head(x)
            return res
        
        res = self.forward_features(x, adapter_id, train)
        res = self.forward_head(res)
        
        return res
        

def vit_base_patch16_224_mos(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'adapter' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model



def vit_base_patch16_224_in21k_mos(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    
    for name, p in model.named_parameters():
        if 'head' in name or 'adapter' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

