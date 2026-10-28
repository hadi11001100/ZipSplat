"""
Background masking of the DA3 tokens, between ZipSplat's encode and decode.

The zipsplat package stays untouched: this script calls ZipSplat.encode, masks the encoded
tokens and hands them to ZipSplat.decode. The masked tokens of all views are packed into one
pseudo-view (B, 1, N_fg, D). The fusion blocks carry no positional encoding, so decode accepts
any token count.

The foreground comes from RMBG-2.0, a token is kept when the average-pooled mask over its patch
exceeds a threshold.

Limits for now:
    - compression=1.0 only: complexity_kmeans seeds from all V * T patches of the images.
    - no TokenBA (ZipSplat.refine): it expects assignments over all V * T tokens.

Usage:
    python -m experiments.test_masking img0.png img1.png ... --out outputs/masking
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from zipsplat import ZipSplat, load_image
from zipsplat.utils import remove_batch_dim

import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation


def get_background_mask(images):
    """Foreground mask [(1, h, w), ...] in [0, 1] of the raw views [(3, h, w), ...].

    RMBG-2.0 (BiRefNet), gated on the HF hub: accept the license and log in first. The soft
    alpha is thresholded in main, after the crop and resize.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default="zipsplat")
    parser.add_argument("--out", type=Path, default=Path("outputs/masking"))
    args = parser.parse_args()

    # Masks first: RMBG is freed before ZipSplat goes to the GPU.
    images = [load_image(p) for p in args.images]
    masks = get_background_mask(images)

    # The wrapper loads the weights and crops the inputs, the network itself is wrapper.model.
    wrapper = ZipSplat(weights=args.weights).cuda().eval()
    model = wrapper.model

    # Same center crop and resize for images and masks (masks as 3 channels, keep channel 0).
    x, _, _ = wrapper.prepare_inputs(images)
    mask, _, _ = wrapper.prepare_inputs([m.expand(3, -1, -1) for m in masks])
    x, mask = x[None], (mask[None, :, 0] > 0.5).float()  # (1, V, 3, H, W), (1, V, H, W)

    with torch.no_grad(), wrapper.autocast():
        # no reference to the full encoding: it is freed before decode
        enc_masked, keep = apply_token_masking(model.encode(x), mask, model.patch_size)
        pred = model.decode(enc_masked, x, compression=1.0)
    pred = remove_batch_dim(pred)

    # Checks: one scene token per kept token, G Gaussians per scene token.
    N_fg = int(keep.sum())
    G = model.gaussian_head.gaussians_per_token
    assert enc_masked["geo_tokens"].shape[2] == N_fg
    assert pred["scene_tokens"].shape[0] == N_fg
    assert pred["gaussians"].shape[0] == N_fg * G
    print(f"kept {N_fg} / {keep.numel()} tokens -> {pred['gaussians'].shape[0]} Gaussians")

    # Render the Gaussians back into the input views, next to the inputs.
    with torch.no_grad():
        rgb, _ = pred["gaussians"].render(pred["cameras"], pred["poses"])  # (V, 3, H, W)
    args.out.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([x[0], rgb.clamp(0, 1)], dim=0)  # inputs in row 1, renders in row 2
    save_image(grid, args.out / "inputs_vs_render.png", nrow=x.shape[1])
    save_image(mask[0, :, None], args.out / "masks.png", nrow=x.shape[1])
    pred["gaussians"].save_ply(args.out / "scene.ply")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
