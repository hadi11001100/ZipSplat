"""
Plain ZipSplat baseline without masking: images -> encode -> decode (optionally TokenBA).

Same outputs as experiments/run_masked_zipsplat.py, saved to
experiments/outputs/baseline_<date>_<time>/, for a direct comparison.

Usage:
    source .venv/bin/activate
    python -m experiments.run_zipsplat experiments/rig/ImageTest2/*.jpg
    python -m experiments.run_zipsplat experiments/rig/ImageTest2/*.jpg --masks experiments/rig/labels
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

from experiments.masking import load_masks, prepare_masks
from zipsplat import ZipSplat, load_image
from zipsplat.utils import remove_batch_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default="zipsplat")
    parser.add_argument("--ba", action="store_true", help="refine with TokenBA (off = fair to masking)")
    parser.add_argument(
        "--masks", type=Path, default=None,
        help="optional YOLO polygon labels, one <stem>.txt per image: adds foreground metrics",
    )
    args = parser.parse_args()

    # One folder per run, next to this script.
    out = Path(__file__).parent / "outputs" / f"baseline_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Masks first: a missing or mis-sized mask fails before ZipSplat loads.
    masks = load_masks(args.masks, args.images) if args.masks else None
    images = [load_image(p) for p in args.images]
    wrapper = ZipSplat(weights=args.weights).cuda().eval()
    model = wrapper.model

    # Same center crop and resize as the masked script, outside the timer like there.
    x, _, _ = wrapper.prepare_inputs(images)  # (V, 3, H, W)
    mask = prepare_masks(wrapper, masks) if masks is not None else None  # (1, V, H, W)

    # ba=False: exactly encode -> decode with compression=1.0, like the masked script without masking.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), wrapper.autocast():
        pred = remove_batch_dim(model(x[None], compression=1.0))
    pred["model_images"], pred["model_cameras"] = x, pred.pop("cameras")
    if args.ba:
        pred = wrapper.refine(pred)
    torch.cuda.synchronize()
    time_s = time.perf_counter() - t0  # encode -> decode (+ TokenBA if --ba)

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
    if mask is not None:
        save_image(mask[0, :, None], out / "masks.png", nrow=V)
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
    metrics_fg = {}
    if mask is not None:
        # Foreground metrics exactly like the masked script.
        m = mask[0, :, None]  # (V, 1, H, W)
        n_fg = m.sum((1, 2, 3))  # (V,) foreground pixels per view
        has_fg = n_fg > 0  # views without foreground have no psnr_fg (would be inf)
        psnr_fg = -10 * ((se * m).sum((1, 2, 3)) / (3 * n_fg.clamp(min=1))).log10()  # (V,)
        ssim_map = torch.stack([
            torch.from_numpy(structural_similarity(
                g.cpu().numpy(), h.cpu().numpy(), win_size=11, gaussian_weights=True,
                channel_axis=0, data_range=1.0, full=True,
            )[1])
            for g, h in zip(gt, hat)
        ]).to(m)  # (V, 3, H, W)
        ssim_fg = (ssim_map * m).sum((1, 2, 3)) / (3 * n_fg.clamp(min=1))  # (V,)
        with torch.no_grad():
            del lpips_fn
            lpips_map_fn = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(gt.device)
            lpips_map = torch.cat([lpips_map_fn(g[None], h[None], normalize=True) for g, h in zip(gt, hat)])
            del lpips_map_fn  # (V, 1, H, W)
        lpips_fg = (lpips_map * m).sum((1, 2, 3)) / n_fg.clamp(min=1)  # (V,)
        fg = lambda t: [v if ok else None for v, ok in zip(t.tolist(), has_fg.tolist())]  # noqa: E731
        mean_fg = lambda t: t[has_fg].mean().item() if has_fg.any() else None  # noqa: E731
        metrics_fg = {
            "psnr_fg": mean_fg(psnr_fg),
            "num_views_empty_fg": int((~has_fg).sum()),
            "psnr_fg_per_view": fg(psnr_fg),
            "ssim_fg": mean_fg(ssim_fg),
            "ssim_fg_per_view": fg(ssim_fg),
            "lpips_fg": mean_fg(lpips_fg),
            "lpips_fg_per_view": fg(lpips_fg),
        }
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
        **metrics_fg,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(
        f"psnr {metrics['psnr']:.2f} | ssim {metrics['ssim']:.3f} | lpips {metrics['lpips']:.3f}"
        + (
            f" | psnr_fg {metrics['psnr_fg']} | ssim_fg {metrics['ssim_fg']} | lpips_fg {metrics['lpips_fg']}"
            if metrics_fg else ""
        )
        + f" | {time_s:.2f} s"
    )
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
