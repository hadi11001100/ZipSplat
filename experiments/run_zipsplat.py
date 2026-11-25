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

from experiments.masking import MaskedTokenBA, load_masks, prepare_masks
from experiments.run_masked_zipsplat import (
    drift_stats, foreground_metrics, free_network, load_colmap_priors, moved_start, select_views,
)
from zipsplat import Camera, ZipSplat, load_image
from zipsplat.token_ba import TokenBA
from zipsplat.utils import remove_batch_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default="zipsplat")
    parser.add_argument("--ba", action="store_true", help="refine with TokenBA")
    parser.add_argument(
        "--masks", type=Path, default=None,
        help="optional YOLO polygon labels, one <stem>.txt per image: adds foreground metrics",
    )
    parser.add_argument("--rounds", type=int, default=0, help="PnP rounds of TokenBA, 0 = same as the masked script")
    parser.add_argument(
        "--holdout", type=int, default=0,
        help="every K-th view (i %% K == K - 1) is a test view: no Gaussians, no BA, 0 = off",
    )
    parser.add_argument("--sparse", type=Path, default=None, help="COLMAP text model as camera priors")
    parser.add_argument("--compression", type=float, default=1.0, help="fraction of the tokens kept, (0, 1]")
    parser.add_argument("--out", type=Path, default=None, help="output folder, default: outputs/baseline_<time>")
    args = parser.parse_args()
    assert 0 < args.compression <= 1, "--compression must be in (0, 1]"
    assert args.masks or not args.holdout, "--holdout needs --masks (foreground pose alignment)"

    # One folder per run, next to this script.
    out = args.out or Path(__file__).parent / "outputs" / f"baseline_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Masks first: a missing or mis-sized mask fails before ZipSplat loads.
    masks = load_masks(args.masks, args.images) if args.masks else None
    images = [load_image(p) for p in args.images]
    wrapper = ZipSplat(weights=args.weights).cuda().eval()
    model = wrapper.model

    # Same center crop and resize as the masked script, outside the timer like there.
    x, offset, scale = wrapper.prepare_inputs(images)  # (V, 3, H, W)
    mask = prepare_masks(wrapper, masks) if masks is not None else None  # (1, V, H, W)

    # Optional camera priors and train / test split, same as the masked script.
    priors = {}
    if args.sparse:
        prior_poses, f, c = load_colmap_priors(args.sparse, [p.name for p in args.images], offset, scale)
        size = torch.tensor([x.shape[-1], x.shape[-2]], dtype=f.dtype, device=f.device).expand(len(x), 2)
        priors = {"poses": prior_poses[None], "cameras": Camera(torch.cat([size, f, c], -1))[None]}  # (1, V)
    idx = torch.arange(len(x), device=x.device)
    is_test = (idx % args.holdout == args.holdout - 1) if args.holdout else torch.zeros_like(idx, dtype=torch.bool)
    tr, te = idx[~is_test], idx[is_test]

    # ba=False: exactly encode -> decode with compression=1.0, like the masked script without masking.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), wrapper.autocast():
        enc = model.encode(x[None], **priors)
        if args.holdout:
            enc = select_views(enc, tr)  # test views give no tokens, their poses still come from cam_token
        pred = model.decode(enc, x[None, tr], compression=args.compression)
        del enc
    net_poses = pred["poses"]  # (1, V), before any BA
    ba_metrics = {}
    if args.ba:
        torch.cuda.synchronize()
        peak_net = torch.cuda.max_memory_allocated()
        del pred["gaussians"]
        free_network(model)
        torch.cuda.reset_peak_memory_stats()
        ba_out = TokenBA(rounds=args.rounds)(
            pred["scene_tokens"], model.gaussian_head, pred["cameras"][:, tr], net_poses[:, tr], x[None, tr],
            assignments=pred["assignments"],
        )
        ba_metrics["peak_mem_ba_gb"] = torch.cuda.max_memory_allocated() / 1e9
        poses = net_poses.data_.clone()  # (1, V, D): refined train poses, network test poses
        poses[:, tr] = ba_out.pop("poses").data_
        pred = {**pred, **ba_out, "poses": type(net_poses)(poses)}
    torch.cuda.synchronize()
    time_s = time.perf_counter() - t0  # encode -> decode (+ TokenBA if --ba)

    if args.ba:
        ba_metrics.update(drift_stats("drift_train", net_poses[0, tr], pred["poses"][0, tr]))
        ba_metrics.update(drift_stats("drift_view0", net_poses[0, :1], pred["poses"][0, :1]))
    if args.holdout:
        # Pose alignment to the fixed scene like the masked script, RGB on the foreground only:
        # the near-constant opacity term of an unmasked scene would trigger the early stop.
        def align_poses(views, start):
            return MaskedTokenBA(mask[:, views], alpha_weight=0.0, rounds=0, joint_token_lr=0.0)(
                pred["scene_tokens"], model.gaussian_head, pred["cameras"][:, views], start, x[None, views]
            )["poses"]

        if not args.ba:
            peak_net = torch.cuda.max_memory_allocated()
            free_network(model)
            torch.cuda.reset_peak_memory_stats()
        align = te if args.ba else idx
        ba_poses = pred["poses"]
        poses = ba_poses.data_.clone()
        poses[:, align] = align_poses(align, ba_poses[:, align]).data_
        pred["poses"] = type(ba_poses)(poses)
        if len(te):
            ba_metrics.update(drift_stats("align_test", net_poses[0, te], pred["poses"][0, te]))
        if args.ba and len(te):
            # Diagnostic: the test poses first follow the train pose correction.
            moved = moved_start(net_poses[0, te], net_poses[0, tr], ba_poses[0, tr])
            moved = align_poses(te, moved[None])
            with torch.no_grad():
                moved_rgb, _ = pred["gaussians"].render(pred["cameras"][:, te], moved)  # (1, V_te, 3, H, W)
    pred = remove_batch_dim(pred)
    pred["model_cameras"] = pred.pop("cameras")

    # Checks: K = r * N scene tokens (all at r = 1), G Gaussians per token.
    V, _, H, W = x.shape
    N = len(tr) * (H // model.patch_size) * (W // model.patch_size)
    K = N if args.compression >= 1.0 else max(1, int(N * args.compression))
    G = model.gaussian_head.gaussians_per_token
    assert pred["scene_tokens"].shape[0] == K
    assert pred["gaussians"].shape[0] == K * G
    print(f"kept {N} / {N} tokens -> {K} scene tokens -> {pred['gaussians'].shape[0]} Gaussians")

    # Render the Gaussians back into the input views, next to the inputs.
    with torch.no_grad():
        rgb, infos = pred["gaussians"].render(pred["model_cameras"], pred["poses"])  # (V, 3, H, W)
    out.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([x, rgb.clamp(0, 1)], dim=0)  # inputs in row 1, renders in row 2
    save_image(grid, out / "inputs_vs_render.png", nrow=V)
    if mask is not None:
        save_image(mask[0, :, None], out / "masks.png", nrow=V)
    pred["gaussians"].save_ply(out / "scene.ply")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9  # before the LPIPS network is loaded
    if args.ba or args.holdout:
        peak_mem_gb = max(peak_mem_gb, peak_net / 1e9)  # the peak was reset after the network
    if args.ba and len(te):
        moved_fg = foreground_metrics(moved_rgb[0], x[te], mask[0, te][:, None])
        ba_metrics.update({f"{k}_test_moved": v for k, v in moved_fg.items()})

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
        alpha_bg_v = (infos["alphas"] * (1 - m)).sum((1, 2, 3)) / (1 - m).sum((1, 2, 3)).clamp(min=1)  # (V,)
        metrics_fg["alpha_bg"] = ((infos["alphas"] * (1 - m)).sum() / (1 - m).sum().clamp(min=1)).item()
        if args.holdout:
            metrics_fg.update(test_views=te.tolist(), pose_aligned_views=align.tolist())
            for name, sel in (("train", tr), ("test", te)):
                ok = sel[has_fg[sel]]
                for key, t in (("psnr_fg", psnr_fg), ("ssim_fg", ssim_fg), ("lpips_fg", lpips_fg), ("alpha_bg", alpha_bg_v)):
                    metrics_fg[f"{key}_{name}"] = t[ok].mean().item() if len(ok) else None
    metrics = {
        "args": vars(args),  # Paths via default=str
        "num_views": V,
        "H": H,
        "W": W,
        "tokens_kept": N,
        "tokens_total": N,
        "keep_ratio": 1.0,
        "scene_tokens": K,
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
        **ba_metrics,
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
