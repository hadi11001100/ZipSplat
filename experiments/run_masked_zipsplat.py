"""
ZipSplat with background-masked tokens: images + mask files -> encode -> token masking -> decode.

The masking itself lives in experiments/masking.py, this script only wires it into ZipSplat and
saves the results to experiments/outputs/masking_<date>_<time>/.

Usage:
    python -m experiments.run_masked_zipsplat experiments/rig/ImageTest2/*.jpg \
        --masks experiments/rig/labels
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torchvision.utils import save_image

from experiments.masking import apply_token_masking, load_masks, prepare_masks
from zipsplat import ZipSplat, load_image
from zipsplat.utils import remove_batch_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default="zipsplat")
    parser.add_argument(
        "--masks", type=Path, required=True, help="YOLO polygon labels, one <stem>.txt per image"
    )
    args = parser.parse_args()

    # One folder per run, next to this script.
    out = Path(__file__).parent / "outputs" / f"masking_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Masks first: a missing or mis-sized mask fails before ZipSplat loads.
    masks = load_masks(args.masks, args.images)
    images = [load_image(p) for p in args.images]

    # The wrapper loads the weights and crops the inputs, the network itself is wrapper.model.
    wrapper = ZipSplat(weights=args.weights).cuda().eval()
    model = wrapper.model

    # Same center crop and resize for images and masks.
    x, _, _ = wrapper.prepare_inputs(images)
    x, mask = x[None], prepare_masks(wrapper, masks)  # (1, V, 3, H, W), (1, V, H, W)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), wrapper.autocast():
        # no reference to the full encoding: it is freed before decode
        enc_masked, keep = apply_token_masking(model.encode(x), mask, model.patch_size)
        pred = model.decode(enc_masked, x, compression=1.0)
    torch.cuda.synchronize()
    time_s = time.perf_counter() - t0  # encode -> masking -> decode
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
    out.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([x[0], rgb.clamp(0, 1)], dim=0)  # inputs in row 1, renders in row 2
    save_image(grid, out / "inputs_vs_render.png", nrow=x.shape[1])
    save_image(mask[0, :, None], out / "masks.png", nrow=x.shape[1])
    pred["gaussians"].save_ply(out / "scene.ply")

    # Metrics: PSNR of the renders on the input views, over all pixels and over the foreground only.
    V = x.shape[1]
    se = (rgb.clamp(0, 1).float() - x[0].float()) ** 2  # (V, 3, H, W)
    m = mask[0, :, None]  # (V, 1, H, W)
    psnr = -10 * se.mean((1, 2, 3)).log10()  # (V,)
    n_fg = m.sum((1, 2, 3))  # (V,) foreground pixels per view
    has_fg = n_fg > 0  # views without foreground have no psnr_fg (would be inf)
    psnr_fg = -10 * ((se * m).sum((1, 2, 3)) / (3 * n_fg.clamp(min=1))).log10()  # (V,)
    metrics = {
        "args": vars(args),  # Paths via default=str
        "num_views": V,
        "H": x.shape[-2],
        "W": x.shape[-1],
        "tokens_kept": N_fg,
        "tokens_total": keep.numel(),
        "keep_ratio": N_fg / keep.numel(),
        "tokens_kept_per_view": keep.view(V, -1).sum(1).tolist(),
        "num_gaussians": pred["gaussians"].shape[0],
        "ply_mb": (out / "scene.ply").stat().st_size / 1e6,
        "time_s": time_s,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "psnr": psnr.mean().item(),
        "psnr_fg": psnr_fg[has_fg].mean().item() if has_fg.any() else None,
        "num_views_empty_fg": int((~has_fg).sum()),
        "psnr_per_view": psnr.tolist(),
        "psnr_fg_per_view": [p if ok else None for p, ok in zip(psnr_fg.tolist(), has_fg.tolist())],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"psnr {metrics['psnr']:.2f} | psnr_fg {metrics['psnr_fg']} | {time_s:.2f} s")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
