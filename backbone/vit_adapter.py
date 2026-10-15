import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
import logging
import os
import copy


class SparseVariationalLinear(nn.Module):
    
    def __init__(self, in_features, out_features, sparse_init=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.q_m = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.q_m, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Frozen previous-task posterior mean.
        self.register_buffer(
            "prior_mu0",
            self.q_m.detach().clone()
        )
        self.kl_loss = torch.zeros(())

    def forward(self, x):
        return F.linear(x, self.q_m, self.bias)

    def conditional_kl(self, e_inv_z):
        
        return 0.5 * e_inv_z * (
            self.q_m - self.prior_mu0
        ).pow(2).sum()

    def reset_kl(self):
        self.kl_loss = torch.zeros(
            (), device=self.q_m.device, dtype=self.q_m.dtype
        )


class Adapter(nn.Module):
    
    def __init__(
        self,
        config=None,
        d_model=None,
        bottleneck=None,
        dropout=0.0,
        init_option="bert",
        adapter_scalar="1.0",
        adapter_layernorm_option="in",
    ):
        super().__init__()

        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        self.scale = (
            nn.Parameter(torch.ones(1))
            if adapter_scalar == "learnable_scalar"
            else float(adapter_scalar)
        )

        self.down_proj = SparseVariationalLinear(
            self.n_embd, self.down_size
        )
        self.non_linear_func = nn.ReLU()
        self.up_proj = SparseVariationalLinear(
            self.down_size, self.n_embd
        )
        self.dropout = dropout

        
        mu0 = 0.1
        lambda0 = 5.0
        var0 = mu0 ** 3 / lambda0

        b20 = math.log1p(var0 / mu0 ** 2)
        b0 = math.sqrt(b20)
        a0 = math.log(mu0) - 0.5 * b20

        self.q_a = nn.Parameter(
            torch.tensor(a0, dtype=torch.float32)
        )
        self.q_b = nn.Parameter(
            torch.tensor(math.log(b0), dtype=torch.float32)
        )

        self.register_buffer(
            "prior_mu",
            torch.tensor(mu0, dtype=torch.float32)
        )
        self.register_buffer(
            "prior_lambda",
            torch.tensor(lambda0, dtype=torch.float32)
        )

        self.register_buffer(
            "prior_mu0_down",
            self.down_proj.q_m.detach().clone()
        )
        self.register_buffer(
            "prior_mu0_up",
            self.up_proj.q_m.detach().clone()
        )

        
        self.z_min = 0.05
        self.inv_z_max = 20.0
        self.var_z_min = 1e-4

        
        self.has_previous_task = False

        self.kl_loss = torch.zeros(())

    def _safe_ab(self):
        
        a = self.q_a.clamp(-5.0, 0.5)
        b = self.q_b.clamp(-4.0, -0.5)
        return a, b

    def _moments(self):
        a, b = self._safe_ab()
        b2 = torch.exp(2.0 * b).clamp(max=0.5)

        
        raw_mean = torch.exp(a + 0.5 * b2)
        raw_second = torch.exp(2.0 * a + 2.0 * b2)

        raw_var = (
            raw_second - raw_mean.pow(2)
        ).clamp_min(self.var_z_min)

        # z = z_min + lognormal variable.
        ez = raw_mean + self.z_min

        
        einvz = torch.rsqrt(
            ez.clamp_min(self.z_min)
        ).pow(2)
        einvz = einvz.clamp(max=self.inv_z_max)

        return ez, raw_var, einvz

    def latent_kl(self):
        
        a, b = self._safe_ab()
        b2 = torch.exp(2.0 * b).clamp(max=0.5)

        mu = self.prior_mu.clamp(1e-3, 10.0)
        lam = self.prior_lambda.clamp(1e-2, 100.0)

        raw_ez = torch.exp(a + 0.5 * b2)
        raw_einvz = torch.exp(-a + 0.5 * b2)

        eq_logq = (
            -a - b
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5
        )

        eq_logp = (
            0.5 * (
                torch.log(lam)
                - math.log(2.0 * math.pi)
            )
            - 1.5 * (a + 0.5 * b2)
            - (lam / (2.0 * mu.pow(2))) *
              (
                  raw_ez
                  + mu.pow(2) * raw_einvz
                  - 2.0 * mu
              )
        )

        return (eq_logq - eq_logp).clamp_min(0.0)

    def compute_kl(self):
        
        latent = self.latent_kl()

        if not self.has_previous_task:
            return latent

        _, _, einvz = self._moments()

        conditional = (
            self.down_proj.conditional_kl(einvz)
            + self.up_proj.conditional_kl(einvz)
        )

        return latent + conditional

    def get_kl(self):
        return self.kl_loss

    def reset_kl(self):
        self.kl_loss = torch.zeros(
            (),
            device=self.q_a.device,
            dtype=self.q_a.dtype
        )

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == "in":
            x_in = self.adapter_layer_norm_before(x)
        else:
            x_in = x

        down = self.down_proj(x_in)
        down = self.non_linear_func(down)
        down = F.dropout(
            down,
            p=self.dropout,
            training=self.training
        )
        up = self.up_proj(down)

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        up = up * self.scale

        self.kl_loss = self.compute_kl()

        return up + residual if add_residual else up

    def update_priors(self, rho=0.05):
        
        with torch.no_grad():
            ez, varz, _ = self._moments()

            mu_hat = ez.clamp(1e-3, 10.0)
            lambda_hat = (
                mu_hat.pow(3)
                / varz.clamp_min(self.var_z_min)
            ).clamp(0.1, 100.0)

            self.prior_mu.mul_(1.0 - rho).add_(
                rho * mu_hat
            )
            self.prior_lambda.mul_(1.0 - rho).add_(
                rho * lambda_hat
            )

            self.prior_mu0_down.copy_(
                self.down_proj.q_m.detach()
            )
            self.prior_mu0_up.copy_(
                self.up_proj.q_m.detach()
            )

            self.down_proj.prior_mu0.copy_(
                self.down_proj.q_m.detach()
            )
            self.up_proj.prior_mu0.copy_(
                self.up_proj.q_m.detach()
            )

            self.has_previous_task = True

    def get_group_scale(self):
        return self._moments()[0].detach()

    def get_group_variance(self):
        return self._moments()[1].detach()

    def get_group_inverse_scale(self):
        return self._moments()[2].detach()

    def get_group_activity(self, threshold=0.01):
        return float(self.get_group_scale() > threshold)


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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None, total_layers=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.total_layers = total_layers
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt: #adapter to all layers    
            self.adaptmlp = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        residual = x
        x_mlp = self.mlp_drop(
            self.act(self.fc1(self.norm2(x)))
        )
        x_mlp = self.drop_path(
            self.mlp_drop(self.fc2(x_mlp))
        )

        if hasattr(self, "adaptmlp"):
            if self.config.ffn_option == "sequential":
                x = self.adaptmlp(
                    x_mlp,
                    add_residual=False
                )
            elif self.config.ffn_option == "parallel":
                adapt_x = self.adaptmlp(
                    residual,
                    add_residual=False
                )
                x = x_mlp + adapt_x
            else:
                raise ValueError(self.config.ffn_option)
        else:
            x = x_mlp

        return residual + x

class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        print("I'm using ViT with SVP adapters.")
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
            config=tuning_config, layer_id=i, total_layers=depth
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

    
    def reset_kl(self):
        for blk in self.blocks:
            if hasattr(blk, "adaptmlp"):
                blk.adaptmlp.reset_kl()

    def get_total_kl(self):
        total = torch.zeros(
            (),
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype
        )
        for blk in self.blocks:
            if hasattr(blk, "adaptmlp"):
                total = total + blk.adaptmlp.get_kl()
        return total

    def get_kl_statistics(self):
        raw = self.get_total_kl()
        groups = [
            blk.adaptmlp
            for blk in self.blocks
            if hasattr(blk, "adaptmlp")
        ]
        n = max(1, len(groups))
        return {
            "raw_kl": raw.detach(),
            "kl_per_group": raw.detach() / n,
            "num_groups": n,
        }

    def get_tailadapt_group_stats(self):
        stats = []
        for i, blk in enumerate(self.blocks):
            if hasattr(blk, "adaptmlp"):
                ad = blk.adaptmlp
                stats.append({
                    "layer": i,
                    "mean_z": float(
                        ad.get_group_scale().cpu()
                    ),
                    "var_z": float(
                        ad.get_group_variance().cpu()
                    ),
                    "inverse_z": float(
                        ad.get_group_inverse_scale().cpu()
                    ),
                    "active": ad.get_group_activity(),
                })
        return stats

    def update_svp_priors(self, rho=0.05):
        # Call ONLY once after a task has finished.
        with torch.no_grad():
            for blk in self.blocks:
                if hasattr(blk, "adaptmlp"):
                    blk.adaptmlp.update_priors(rho=rho)

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x,)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

def vit_base_patch16_224_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
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
    
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
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
    
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model