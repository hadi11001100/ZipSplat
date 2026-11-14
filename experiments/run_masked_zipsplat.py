"""
ZipSplat with background-masked tokens: images + mask files -> encode -> token masking -> decode.

The masking itself lives in experiments/masking.py, this script only wires it into ZipSplat and
saves the results to experiments/outputs/masking_<date>_<time>/.

Usage:
    python -m experiments.run_masked_zipsplat experiments/rig/ImageTest2/*.jpg \
        --masks experiments/rig/labels
    python -m experiments.run_masked_zipsplat experiments/rig/ImageTest2/*.jpg \
        --masks experiments/rig/labels --ba
    python -m experiments.run_masked_zipsplat experiments/rig/ImageTest2/*.jpg \
        --masks experiments/rig/labels --ba --holdout 8
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

from experiments.masking import MaskedTokenBA, apply_token_masking, load_masks, prepare_masks
from zipsplat import ZipSplat, load_image
from zipsplat.utils import remove_batch_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default="zipsplat")
    parser.add_argument(
        "--masks", type=Path, required=True, help="YOLO polygon labels, one <stem>.txt per image"
    )
    parser.add_argument(
        "--ba", action="store_true", help="refine with MaskedTokenBA (rounds=0, same as the baseline)"
    )
    parser.add_argument(
        "--holdout", type=int, default=0,
        help="every K-th view (i %% K == K - 1) is a test view: no Gaussians, no BA, 0 = off",
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

    # Holdout: test views give no tokens (mask 0 for the token selection) and no BA loss. Their
    # poses still come from the network (cam_token of the full views). View 0 stays a train view.
    idx = torch.arange(x.shape[1], device=x.device)
    is_test = (idx % args.holdout == args.holdout - 1) if args.holdout else torch.zeros_like(idx, dtype=torch.bool)
    tr, te = idx[~is_test], idx[is_test]
    mask_tok = mask * (~is_test)[None, :, None, None]  # (1, V, H, W)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), wrapper.autocast():
        # no reference to the full encoding: it is freed before decode
        enc_masked, keep = apply_token_masking(model.encode(x), mask_tok, model.patch_size)
        pred = model.decode(enc_masked, x, compression=1.0)
    if args.ba:
        # batched (B == 1), outside autocast like ZipSplat.refine
        ba_out = MaskedTokenBA(mask[:, tr], rounds=0)(
            pred["scene_tokens"], model.gaussian_head, pred["cameras"][:, tr], pred["poses"][:, tr], x[:, tr]
        )
        poses = pred["poses"].data_.clone()  # (1, V, D): refined train poses, network test poses
        poses[:, tr] = ba_out.pop("poses").data_
        pred = {**pred, **ba_out, "poses": type(pred["poses"])(poses)}  # refined gaussians, scene_tokens
    torch.cuda.synchronize()
    time_s = time.perf_counter() - t0  # encode -> masking -> decode (+ MaskedTokenBA if --ba)
    if len(te):
        # Fair test evaluation (with and without BA alike): align only the test poses to the fixed
        # scene, joint_token_lr=0 keeps the scene tokens unchanged. Not timed, it is evaluation.
        test_poses = MaskedTokenBA(mask[:, te], rounds=0, joint_token_lr=0.0)(
            pred["scene_tokens"], model.gaussian_head, pred["cameras"][:, te], pred["poses"][:, te], x[:, te]
        )["poses"]
        poses = pred["poses"].data_.clone()
        poses[:, te] = test_poses.data_
        pred["poses"] = type(pred["poses"])(poses)
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
        rgb, infos = pred["gaussians"].render(pred["cameras"], pred["poses"])  # (V, 3, H, W)
    out.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([x[0], rgb.clamp(0, 1)], dim=0)  # inputs in row 1, renders in row 2
    save_image(grid, out / "inputs_vs_render.png", nrow=x.shape[1])
    save_image(mask[0, :, None], out / "masks.png", nrow=x.shape[1])
    pred["gaussians"].save_ply(out / "scene.ply")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9  # before the LPIPS network is loaded

    # Metrics: PSNR of the renders on the input views, over all pixels and over the foreground only.
    V = x.shape[1]
    se = (rgb.clamp(0, 1).float() - x[0].float()) ** 2  # (V, 3, H, W)
    m = mask[0, :, None]  # (V, 1, H, W)
    psnr = -10 * se.mean((1, 2, 3)).log10()  # (V,)
    n_fg = m.sum((1, 2, 3))  # (V,) foreground pixels per view
    has_fg = n_fg > 0  # views without foreground have no psnr_fg (would be inf)
    psnr_fg = -10 * ((se * m).sum((1, 2, 3)) / (3 * n_fg.clamp(min=1))).log10()  # (V,)

    # SSIM and LPIPS, same settings as splatfactory/models/metrics.py. The foreground variants
    # average the per-pixel SSIM / LPIPS maps over the mask.
    hat, gt = rgb.clamp(0, 1).float(), x[0].float()  # (V, 3, H, W)
    ssim, ssim_map = zip(*[
        structural_similarity(
            g.cpu().numpy(), h.cpu().numpy(), win_size=11, gaussian_weights=True,
            channel_axis=0, data_range=1.0, full=True,
        )
        for g, h in zip(gt, hat)
    ])
    ssim = torch.tensor(ssim)  # (V,)
    ssim_map = torch.stack([torch.from_numpy(s) for s in ssim_map]).to(m)  # (V, 3, H, W)
    ssim_fg = (ssim_map * m).sum((1, 2, 3)) / (3 * n_fg.clamp(min=1))  # (V,)
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(gt.device)
        lpips_val = torch.cat([lpips_fn(g[None], h[None], normalize=True).flatten() for g, h in zip(gt, hat)])
        del lpips_fn
        lpips_map_fn = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(gt.device)
        lpips_map = torch.cat([lpips_map_fn(g[None], h[None], normalize=True) for g, h in zip(gt, hat)])
        del lpips_map_fn  # (V, 1, H, W)
    lpips_fg = (lpips_map * m).sum((1, 2, 3)) / n_fg.clamp(min=1)  # (V,)
    # Mean rendered opacity outside the mask: Gaussians leaking into the background (halo).
    n_bg = (1 - m).sum()
    alpha_bg = ((infos["alphas"] * (1 - m)).sum() / n_bg.clamp(min=1)).item()
    fg = lambda t: [v if ok else None for v, ok in zip(t.tolist(), has_fg.tolist())]  # noqa: E731
    # Train / test split of the foreground metrics (views without foreground left out).
    alpha_bg_v = (infos["alphas"] * (1 - m)).sum((1, 2, 3)) / (1 - m).sum((1, 2, 3)).clamp(min=1)  # (V,)
    split = {}
    for name, sel in (("train", tr), ("test", te)):
        ok = sel[has_fg[sel]]
        for key, t in (("psnr_fg", psnr_fg), ("ssim_fg", ssim_fg), ("lpips_fg", lpips_fg), ("alpha_bg", alpha_bg_v)):
            split[f"{key}_{name}"] = t[ok].mean().item() if len(ok) else None
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
        "peak_mem_gb": peak_mem_gb,
        "psnr": psnr.mean().item(),
        "psnr_fg": psnr_fg[has_fg].mean().item() if has_fg.any() else None,
        "num_views_empty_fg": int((~has_fg).sum()),
        "psnr_per_view": psnr.tolist(),
        "psnr_fg_per_view": [p if ok else None for p, ok in zip(psnr_fg.tolist(), has_fg.tolist())],
        "ssim": ssim.mean().item(),
        "ssim_fg": ssim_fg[has_fg].mean().item() if has_fg.any() else None,
        "ssim_per_view": ssim.tolist(),
        "ssim_fg_per_view": fg(ssim_fg),
        "lpips": lpips_val.mean().item(),
        "lpips_fg": lpips_fg[has_fg].mean().item() if has_fg.any() else None,
        "lpips_per_view": lpips_val.tolist(),
        "lpips_fg_per_view": fg(lpips_fg),
        "alpha_bg": alpha_bg,
        **({"test_views": te.tolist(), **split} if args.holdout else {}),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(
        f"psnr {metrics['psnr']:.2f} | psnr_fg {metrics['psnr_fg']} | ssim {metrics['ssim']:.3f}"
        f" | ssim_fg {metrics['ssim_fg']} | lpips {metrics['lpips']:.3f} | lpips_fg {metrics['lpips_fg']}"
        f" | alpha_bg {alpha_bg:.4f}"
        f" | {time_s:.2f} s"
    )
    if args.holdout:
        print(" | ".join(f"{k} {v:.4f}" for k, v in split.items() if v is not None))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
