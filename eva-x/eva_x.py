"""
EVA-X backbone for chest X-ray (from https://github.com/hustvl/EVA-X).
Use as feature extractor: forward(x) returns feature vector (no classification head).
Requires: timm (with Eva), torch.
"""

import os
import torch
import torch.nn as nn

try:
    from timm.models.eva import Eva
    from timm.layers import resample_abs_pos_embed, resample_patch_embed
except ImportError:
    Eva = None
    resample_abs_pos_embed = resample_patch_embed = None


def checkpoint_filter_fn(state_dict, model, interpolation='bicubic', antialias=True):
    """Convert patch embedding / pos embed for loading EVA-X checkpoint."""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict
    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue
        if 'rope' in k:
            continue
        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v, (H, W), interpolation=interpolation, antialias=antialias, verbose=True
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')
        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                k = k.replace('norm', 'fc_norm')
            else:
                continue
        out_dict[k] = v
    return out_dict


class EVA_X(Eva if Eva is not None else nn.Module):
    """EVA-X backbone (feature extractor only)."""

    def __init__(self, **kwargs):
        if Eva is None:
            raise ImportError("timm is required for EVA-X. Install with: pip install timm>=0.9.0")
        # Feature extractor only: no classification head needed.
        kwargs['num_classes'] = 0
        super().__init__(**kwargs)
        # fc_norm is only called in forward_head(), which we never invoke.
        # Replacing it with Identity eliminates unused-parameter DDP errors
        # when training with DistributedDataParallel.
        self.fc_norm = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        """Return feature vector [B, C] for backbone use (global average pool over sequence)."""
        x = self.forward_features(x)   # [B, N, C]
        x = x.mean(dim=1)               # [B, C]
        return x


def _load_evax_tiny(pretrained_path, img_size=224):
    grid = img_size // 16  # patch_size=16
    model = EVA_X(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(grid, grid),
    )
    ckpt = checkpoint_filter_fn(torch.load(pretrained_path, map_location='cpu', weights_only=False), model)
    model.load_state_dict(ckpt, strict=False)
    return model


def _load_evax_small(pretrained_path, img_size=224):
    grid = img_size // 16  # patch_size=16
    model = EVA_X(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(grid, grid),
    )
    ckpt = checkpoint_filter_fn(torch.load(pretrained_path, map_location='cpu', weights_only=False), model)
    model.load_state_dict(ckpt, strict=False)
    return model


def _load_evax_base(pretrained_path, img_size=224):
    grid = img_size // 16  # patch_size=16
    model = EVA_X(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(grid, grid),
    )
    ckpt = checkpoint_filter_fn(torch.load(pretrained_path, map_location='cpu', weights_only=False), model)
    model.load_state_dict(ckpt, strict=False)
    return model


def eva_x_tiny_patch16(pretrained=False, img_size=224):
    if not pretrained:
        raise ValueError("EVA-X requires pretrained=True and a path to the checkpoint.")
    return _load_evax_tiny(pretrained, img_size=img_size)


def eva_x_small_patch16(pretrained=False, img_size=224):
    if not pretrained:
        raise ValueError("EVA-X requires pretrained=True and a path to the checkpoint.")
    return _load_evax_small(pretrained, img_size=img_size)


def eva_x_base_patch16(pretrained=False, img_size=224):
    if not pretrained:
        raise ValueError("EVA-X requires pretrained=True and a path to the checkpoint.")
    return _load_evax_base(pretrained, img_size=img_size)


# Map short names -> (loader_fn, default_ckpt_filename)
EVA_X_REGISTRY = {
    'eva_x_tiny': (eva_x_tiny_patch16, 'eva_x_tiny_patch16_merged520k_mim.pt'),
    'eva_x_small': (eva_x_small_patch16, 'eva_x_small_patch16_merged520k_mim.pt'),
    'eva_x_base': (eva_x_base_patch16, 'eva_x_base_patch16_merged520k_mim.pt'),
}


def get_evax_backbone(model_name, ckpt_path, img_size=224, verbose=True):
    """
    Load EVA-X backbone for use as feature extractor (output = feature vector, no head).
    model_name: one of 'eva_x_tiny', 'eva_x_small', 'eva_x_base'
    ckpt_path: directory containing the .pt file, or full path to .pt file.
    img_size: input image size (e.g. 224 or 512). Checkpoint is 224; pos_embed/patch_embed are resampled if different.
    """
    if model_name not in EVA_X_REGISTRY:
        raise ValueError(f"Unknown EVA-X model: {model_name}. Choose from {list(EVA_X_REGISTRY.keys())}")
    loader_fn, default_fname = EVA_X_REGISTRY[model_name]
    ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_path)))
    if os.path.isfile(ckpt_path):
        path = ckpt_path
    else:
        path = os.path.join(ckpt_path, default_fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"EVA-X checkpoint not found: {path}\n"
            "Download from https://huggingface.co/MapleF/eva_x (e.g. eva_x_small_patch16_merged520k_mim.pt)"
        )
    if verbose:
        print(f"[EVA-X] Loading {model_name} from {path} (img_size={img_size})")
    return loader_fn(pretrained=path, img_size=int(img_size))
