"""
Plain ZipSplat baseline without masking: images -> encode -> decode (optionally TokenBA).

Same outputs as experiments/run_masked_zipsplat.py, saved to
experiments/outputs/baseline_<date>_<time>/, for a direct comparison.

Usage:
    source .venv/bin/activate
    python -m experiments.run_zipsplat assets/examples/office/*.png
    python -m experiments.run_zipsplat assets/examples/office/*.png --ba
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import lpips
import torch
from skimage.metrics import structural_similarity
from torchvision.utils import save_image

from zipsplat import ZipSplat, load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default="zipsplat")
    parser.add_argument("--ba", action="store_true", help="refine with TokenBA (off = fair to masking)")
    args = parser.parse_args()

    # One folder per run, next to this script.
    out = Path(__file__).parent / "outputs" / f"baseline_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    images = [load_image(p) for p in args.images]
    wrapper = ZipSplat(weights=args.weights).cuda().eval()
    model = wrapper.model

    # ba=False: exactly encode -> decode with compression=1.0, like the masked script without masking.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = wrapper(images, compression=1.0, ba=args.ba)
    torch.cuda.synchronize()
    time_s = time.perf_counter() - t0  # crop/resize + encode -> decode (+ TokenBA if --ba)
    x = pred["model_images"]  # (V, 3, H, W), same crop and resize as the masked script

    # Checks: every token is kept, G Gaussians per token.
    V, _, H, W = x.shape
    N = V * (H // model.patch_size) * (W // model.patch_size)
    G = model.gaussian_head.gaussians_per_token
    assert pred["scene_tokens"].shape[0] == N
    assert pred["gaussians"].shape[0] == N * G
    print(f"kept {N} / {N} tokens -> {pred['gaussians'].shape[0]} Gaussians")

    # Render the Gaussians back into the input views, next to the inputs.
    with torch.no_grad():
        rgb, _ = pred["gaussians"].render(pred["model_cameras"], pred["poses"])  # (V, 3, H, W)
    out.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([x, rgb.clamp(0, 1)], dim=0)  # inputs in row 1, renders in row 2
    save_image(grid, out / "inputs_vs_render.png", nrow=V)
    pred["gaussians"].save_ply(out / "scene.ply")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9  # before the LPIPS network is loaded

    # Metrics: PSNR of the renders on the input views, same keys as the masked script.
    se = (rgb.clamp(0, 1).float() - x.float()) ** 2  # (V, 3, H, W)
    psnr = -10 * se.mean((1, 2, 3)).log10()  # (V,)

    # SSIM and LPIPS, same settings as splatfactory/models/metrics.py.
    hat, gt = rgb.clamp(0, 1).float(), x.float()  # (V, 3, H, W)
    ssim = torch.tensor([
        structural_similarity(
            g.cpu().numpy(), h.cpu().numpy(), win_size=11, gaussian_weights=True,
            channel_axis=0, data_range=1.0,
        )
        for g, h in zip(gt, hat)
    ])  # (V,)
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(gt.device)
        lpips_val = torch.cat([lpips_fn(g[None], h[None], normalize=True).flatten() for g, h in zip(gt, hat)])
    metrics = {
        "args": vars(args),  # Paths via default=str
        "num_views": V,
        "H": H,
        "W": W,
        "tokens_kept": N,
        "tokens_total": N,
        "keep_ratio": 1.0,
        "num_gaussians": pred["gaussians"].shape[0],
        "ply_mb": (out / "scene.ply").stat().st_size / 1e6,
        "time_s": time_s,
        "peak_mem_gb": peak_mem_gb,
        "psnr": psnr.mean().item(),
        "psnr_per_view": psnr.tolist(),
        "ssim": ssim.mean().item(),
        "ssim_per_view": ssim.tolist(),
        "lpips": lpips_val.mean().item(),
        "lpips_per_view": lpips_val.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(
        f"psnr {metrics['psnr']:.2f} | ssim {metrics['ssim']:.3f} | lpips {metrics['lpips']:.3f}"
        f" | {time_s:.2f} s"
    )
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
