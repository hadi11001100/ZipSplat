"""
Background masking of the DA3 tokens, between ZipSplat's encode and decode.

Module only, the pipeline lives in experiments/run_masked_zipsplat.py. The masked tokens of all
views are packed into one pseudo-view (B, 1, N_fg, D). The fusion blocks carry no positional
encoding, so decode accepts any token count.

The foreground comes from RMBG-2.0, a token is kept when the average-pooled mask over its patch
exceeds a threshold.

Limits for now:
    - compression=1.0 only: complexity_kmeans seeds from all V * T patches of the images.
    - no TokenBA (ZipSplat.refine): it expects assignments over all V * T tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation


def get_background_mask(images):
    """Foreground mask [(1, h, w), ...] in [0, 1] of the raw views [(3, h, w), ...].

    RMBG-2.0 (BiRefNet), gated on the HF hub: accept the license and log in first. The soft
    alpha is thresholded in prepare_masks, after the crop and resize.
    """

    rmbg = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
    rmbg = rmbg.cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]

    masks = []
    with torch.no_grad():
        for im in images:
            h, w = im.shape[-2:]
            x = F.interpolate(im[None].cuda(), size=(1024, 1024), mode="bilinear")
            alpha = rmbg((x - mean) / std)[-1].sigmoid()  # (1, 1, 1024, 1024)
            alpha = F.interpolate(alpha.float(), size=(h, w), mode="bilinear")
            masks.append(alpha[0].clamp(0, 1).cpu())

    del rmbg
    torch.cuda.empty_cache()
    return masks


def prepare_masks(wrapper, masks, threshold=0.5):
    """Binary masks (1, V, H, W) with the same center crop and resize as the model images.

    Args:
        wrapper: ZipSplat wrapper, its prepare_inputs crops and resizes the images.
        masks: Soft foreground masks [(1, h, w), ...] from get_background_mask.
        threshold: Minimum alpha of a pixel to count as foreground.
    """
    # prepare_inputs expects RGB: masks as 3 channels, keep channel 0.
    mask, _, _ = wrapper.prepare_inputs([m.expand(3, -1, -1) for m in masks])
    return (mask[None, :, 0] > threshold).float()


def apply_token_masking(enc, mask, patch_size, threshold=0.5):
    """Keep only the foreground tokens of the encoded views.

    Args:
        enc: ZipSplat.encode output: layer_tokens [(B, V, T, D)], geo_tokens (B, V, T, D),
            color_tokens (B, V, T, D), cam_token (B, V, 2 * D).
        mask: Foreground mask (B, V, H, W) in {0, 1}, same crop and size as the model images.
        patch_size: Pixels per token side, T = (H / patch_size) * (W / patch_size).
        threshold: Minimum foreground fraction of a patch to keep its token.

    Returns:
        enc_masked: enc with the token tensors as (B, 1, N_fg, D), cam_token untouched.
        keep: Foreground flag per token (V * T,), in token order (view, row-major patch).
    """
    B, V, T, D = enc["geo_tokens"].shape
    assert B == 1, "N_fg differs per scene: one scene at a time"
    assert mask.shape[-2] * mask.shape[-1] == T * patch_size**2

    # Foreground fraction per patch: (V, 1, H / p, W / p), flattened in token order.
    frac = F.avg_pool2d(mask[0, :, None].float(), kernel_size=patch_size, stride=patch_size)
    keep = (frac.flatten() > threshold).to(enc["geo_tokens"].device)
    assert keep.numel() == V * T

    select = lambda t: t.reshape(B, V * T, D)[:, keep].unsqueeze(1)  # noqa: E731
    enc_masked = {
        "layer_tokens": [select(t) for t in enc["layer_tokens"]],
        "geo_tokens": select(enc["geo_tokens"]),
        "color_tokens": select(enc["color_tokens"]),
        "cam_token": enc["cam_token"],  # the poses still read the full views
    }
    return enc_masked, keep

