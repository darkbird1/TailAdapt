# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
import logging
import copy


class SparseVariationalLinear(nn.Module):
    
    def __init__(self, in_features, out_features, sparse_init=True,
                 shared_q_a=None, shared_q_b=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        
        self.q_m = nn.Parameter(torch.zeros(out_features, in_features))
        self.q_log_s = nn.Parameter(torch.zeros(out_features, in_features))


        init_mu = 0.01
        init_lambda = 5.0
        init_var = (init_mu ** 3) / init_lambda
        init_b_sq = math.log1p(init_var / (init_mu ** 2))
        init_b = math.sqrt(init_b_sq)
        init_a = math.log(init_mu) - 0.5 * init_b_sq

        if shared_q_a is None:
            self.q_a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        else:
            self.q_a = shared_q_a

        if shared_q_b is None:
            self.q_b = nn.Parameter(torch.tensor(math.log(init_b + 1e-12), dtype=torch.float32))
        else:
            self.q_b = shared_q_b


        nn.init.zeros_(self.q_m)
        nn.init.zeros_(self.q_log_s)

        self.bias = nn.Parameter(torch.zeros(out_features))

        
        self.register_buffer('prior_mu', torch.tensor(init_mu, dtype=torch.float32))
        self.register_buffer('prior_lambda', torch.tensor(init_lambda, dtype=torch.float32))
        self.register_buffer('prior_mu0', torch.zeros(out_features, in_features))

        self.kl_loss = 0.0

    def reset_kl(self):
        self.kl_loss = 0.0

    def _moments(self):
        """Moments of LogNormal(a,b^2)."""
        b_sq = torch.exp(2.0 * self.q_b)
        mean_z = torch.exp(self.q_a + 0.5 * b_sq)
        var_z = (torch.exp(b_sq) - 1.0) * torch.exp(2.0 * self.q_a + b_sq)
        return b_sq, mean_z, var_z

    def conditional_kl(self):
        """E_q(z) KL[q(W|z)||p(W|z)] for diagonal S."""
        b_sq = torch.exp(2.0 * self.q_b)
        e_inv_z = torch.exp(-self.q_a + 0.5 * b_sq)

        s_sq = torch.exp(2.0 * self.q_log_s)
        mean_term = e_inv_z * (self.q_m - self.prior_mu0).pow(2)
        var_term = s_sq - 1.0 - 2.0 * self.q_log_s

        
        return 0.5 * (mean_term + var_term).mean()

    def latent_kl(self):
        """Closed-form KL surrogate KL[LogNormal(a,b^2) || IG(mu,lambda)]."""
        eps = 1e-8
        b_sq = torch.exp(2.0 * self.q_b).clamp_min(eps)
        mu = self.prior_mu.clamp_min(eps)
        lam = self.prior_lambda.clamp_min(eps)

        e_z = torch.exp(self.q_a + 0.5 * b_sq)
        e_inv_z = torch.exp(-self.q_a + 0.5 * b_sq)

        
        
        kl = (
            0.5 * self.q_a
            - 0.5 * torch.log(b_sq)
            - 0.5
            - 0.5 * torch.log(lam)
            + 0.5 * lam / (mu ** 2) * (
                e_z + (mu ** 2) * e_inv_z - 2.0 * mu
            )
        )

        return kl.clamp_min(0.0)

    def compute_kl(self, z=None, include_latent=True):
       
        kl = self.conditional_kl()
        if include_latent:
            kl = kl + self.latent_kl()
        return kl

    def forward(self, x):
        if self.training:
            eps_z = torch.randn((), device=self.q_a.device, dtype=self.q_a.dtype)
            z = torch.exp(self.q_a + torch.exp(self.q_b) * eps_z)
            z = z.clamp(min=1e-6, max=10.0)

            eps_w = torch.randn_like(self.q_m)
            sigma = torch.exp(self.q_log_s)
            weight = self.q_m + torch.sqrt(z) * sigma * eps_w
        else:
            # Posterior mean for deterministic inference.
            z = torch.exp(self.q_a).clamp(min=1e-6, max=10.0)
           
            weight = self.q_m

        return F.linear(x, weight, self.bias)



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

        # One block-level latent z shared by both projections.
        init_mu = 0.01
        init_lambda = 5.0
        init_var = (init_mu ** 3) / init_lambda
        init_b_sq = math.log1p(init_var / (init_mu ** 2))
        init_b = math.sqrt(init_b_sq)
        init_a = math.log(init_mu) - 0.5 * init_b_sq

        shared_q_a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        shared_q_b = nn.Parameter(torch.tensor(math.log(init_b + 1e-12), dtype=torch.float32))

        self.down_proj = SparseVariationalLinear(
            self.n_embd, self.down_size,
            shared_q_a=shared_q_a, shared_q_b=shared_q_b)
        self.non_linear_func = nn.ReLU()
        self.up_proj = SparseVariationalLinear(
            self.down_size, self.n_embd,
            shared_q_a=shared_q_a, shared_q_b=shared_q_b)

        self.dropout = dropout

        if init_option == "lora":
            with torch.no_grad():
                nn.init.zeros_(self.down_proj.q_m)
                nn.init.zeros_(self.up_proj.q_m)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def reset_kl(self):
        self.down_proj.reset_kl()
        self.up_proj.reset_kl()

    def get_kl(self):
        # Conditional KL for both matrices + ONE shared latent KL.
        return (
            self.down_proj.conditional_kl()
            + self.up_proj.conditional_kl()
            + self.down_proj.latent_kl()
        )

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

        # attn = (q @ k.transpose(-2, -1)) * self.scale
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

    def forward(self, x, adapt=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt)

        x = residual + x
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        print("I'm using ViT with SVP adapters (EASE).")
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
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in range(depth)]
            )
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)
        
        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()

    # --- SVP Utilities ---
    def reset_kl(self):
        """Reset per-forward KL bookkeeping."""
        for adapt in self.cur_adapter:
            if hasattr(adapt, 'reset_kl'):
                adapt.reset_kl()

    def get_total_kl(self):
        """Return the current-task hierarchical KL."""
        kl = None
        for adapt in self.cur_adapter:
            if hasattr(adapt, 'get_kl'):
                value = adapt.get_kl()
                kl = value if kl is None else kl + value
        if kl is None:
            return torch.tensor(0.0, device=self._device)
        return kl

    def update_svp_priors(self):
        """Moment-match q(z) to IG and apply tempered posterior-to-prior update."""
        rho = 0.05
        with torch.no_grad():
            for adapt in self.cur_adapter:
                if not isinstance(adapt, Adapter):
                    continue

                # The latent z is shared by down/up, so update it once.
                layer = adapt.down_proj
                b_sq = torch.exp(2.0 * layer.q_b)
                mean_z = torch.exp(layer.q_a + 0.5 * b_sq)
                var_z = (torch.exp(b_sq) - 1.0) * torch.exp(
                    2.0 * layer.q_a + b_sq)
                hat_mu = mean_z.clamp_min(1e-8)
                hat_lambda = (hat_mu ** 3) / var_z.clamp_min(1e-8)

                for proj in (adapt.down_proj, adapt.up_proj):
                    proj.prior_mu.copy_((1.0 - rho) * proj.prior_mu + rho * hat_mu)
                    proj.prior_lambda.copy_((1.0 - rho) * proj.prior_lambda + rho * hat_lambda)
                    proj.prior_mu0.copy_((1.0 - rho) * proj.prior_mu0 + rho * proj.q_m)

    def _inherit_tempered_prior(self, source_adapter):
        """Initialize the new adapter from the previous task's tempered prior."""
        if source_adapter is None:
            return

        with torch.no_grad():
            for new_adapt, old_adapt in zip(self.cur_adapter, source_adapter):
                for new_proj, old_proj in [
                    (new_adapt.down_proj, old_adapt.down_proj),
                    (new_adapt.up_proj, old_adapt.up_proj),
                ]:
                    new_proj.prior_mu.copy_(old_proj.prior_mu)
                    new_proj.prior_lambda.copy_(old_proj.prior_lambda)
                    new_proj.prior_mu0.copy_(old_proj.prior_mu0)

                
                new_adapt.down_proj.q_m.copy_(new_adapt.down_proj.prior_mu0)
                new_adapt.up_proj.q_m.copy_(new_adapt.up_proj.prior_mu0)

                
                mu = new_adapt.down_proj.prior_mu.clamp_min(1e-8)
                lam = new_adapt.down_proj.prior_lambda.clamp_min(1e-8)
                var = (mu ** 3 / lam).clamp_min(1e-12)
                b_sq = torch.log1p(var / (mu ** 2)).clamp_min(1e-8)
                a = torch.log(mu) - 0.5 * b_sq
                b = torch.sqrt(b_sq).clamp_min(1e-6)
                new_adapt.down_proj.q_a.copy_(a)
                new_adapt.down_proj.q_b.copy_(torch.log(b))

                
                new_adapt.down_proj.q_log_s.zero_()
                new_adapt.up_proj.q_log_s.zero_()

    # ---------------------

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

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True
        
    def get_new_adapter(self):
        config = self.config
        self.cur_adapter = nn.ModuleList()

        if config.ffn_adapt:
            for i in range(len(self.blocks)):
                adapter = Adapter(
                    self.config,
                    dropout=0.1,
                    bottleneck=config.ffn_num,
                    init_option=config.ffn_adapter_init_option,
                    adapter_scalar=config.ffn_adapter_scalar,
                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                ).to(self._device)
                self.cur_adapter.append(adapter)

            
            source = getattr(self, '_pending_prior_source', None)
            if source is not None:
                self._inherit_tempered_prior(source)
                self._pending_prior_source = None

            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")

    def add_adapter_to_list(self):
        
        source = copy.deepcopy(self.cur_adapter)
        self.reset_kl()
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self._pending_prior_source = source
        self.get_new_adapter()

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)            

        for idx, blk in enumerate(self.blocks):
            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x, self.cur_adapter[idx])
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    def forward_test(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        features = []
        
        if use_init_ptm:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            features.append(x)
        
        for i in range(len(self.adapter_list)):
            x = copy.deepcopy(x_init)
            for j in range(len(self.blocks)):
                adapt = self.adapter_list[i][j]
                x = self.blocks[j](x, adapt)
            x = self.norm(x)
            features.append(x)
        
        x = copy.deepcopy(x_init)
        for i in range(len(self.blocks)):
            adapt = self.cur_adapter[i]
            x = self.blocks[i](x, adapt)
        x = self.norm(x)
        features.append(x)
        
        return features

    def forward(self, x, test=False, use_init_ptm=False):
        if not test:
            output = self.forward_train(x)
        else:
            features = self.forward_test(x, use_init_ptm)
            output = torch.Tensor().to(features[0].device)
            for x in features:
                cls = x[:, 0, :]
                output = torch.cat((
                    output,
                    cls
                ), dim=1)

        return output

    def forward_proto(self, x, adapt_index):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        # the init_PTM's feature
        if adapt_index == -1:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            output = x[:, 0, :]
            return output

        i = adapt_index
        x = copy.deepcopy(x_init)
        for j in range(len(self.blocks)):
            if i < len(self.adapter_list):
                adapt = self.adapter_list[i][j]
            else:
                adapt = self.cur_adapter[j]
            x = self.blocks[j](x, adapt)
        x = self.norm(x)
        output = x[:, 0, :]
        
        return output
        

def vit_base_patch16_224_ease(pretrained=False, **kwargs):
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
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

def vit_base_patch16_224_in21k_ease(pretrained=False, **kwargs):
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
    print(msg)

    
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model