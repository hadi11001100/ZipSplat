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
    python -m experiments.run_masked_zipsplat experiments/rig/ImageTest2/*.jpg \
        --masks experiments/rig/labels --sparse experiments/sparse/0 --compression 0.25
"""

import argparse
import gc
import json
import time
from functools import partial
from datetime import datetime
from pathlib import Path

import lpips
import torch
from skimage.metrics import structural_similarity
from torchvision.utils import save_image

from experiments.masking import (
    MaskedTokenBA, apply_token_masking, load_masks, masked_cluster, prepare_masks,
)
from zipsplat import Camera, Pose, ZipSplat, load_image
from zipsplat.pose import quaternion_to_matrix
from zipsplat.token_ba import TokenBA
from zipsplat.utils import remove_batch_dim


def load_colmap_priors(sparse, names, offset, scale):
    """COLMAP text model -> camera priors for the backbone, in the model input frame.

    Like the training data (splatfactory/datasets/multi_view_dataset.py): c2w OpenCV poses
    relative to view 0, translations divided by the median distance of the sparse points from
    view 0 (training: of the depth points). Lens distortion is ignored.

    Args:
        sparse: Folder with cameras.txt, images.txt, points3D.txt.
        names: Image file names (V,), in input order.
        offset, scale: (V, 2) crop offset and scale from ZipSplat.prepare_inputs.

    Returns:
        c2w poses (V,) and the focal lengths and principal points (V, 2) in model pixels.
    """
    lines = lambda f: [l.split() for l in (sparse / f).read_text().splitlines() if l and l[0] != "#"]  # noqa: E731
    cams = {c[0]: [float(v) for v in c[4:8]] for c in lines("cameras.txt")}  # PINHOLE/OPENCV: fx fy cx cy
    by_name = {l[9]: l for l in lines("images.txt")[::2]}  # every 2nd line is the POINTS2D list
    missing = [n for n in names if n not in by_name]
    assert not missing, f"not in {sparse / 'images.txt'}: {missing}"
    rows = [by_name[n] for n in names]
    q = torch.tensor([[float(v) for v in r[1:5]] for r in rows])  # (V, 4) wxyz, world-to-cam
    t = torch.tensor([[float(v) for v in r[5:8]] for r in rows])  # (V, 3)
    w2c = Pose.from_Rt(quaternion_to_matrix(q[:, [1, 2, 3, 0]]), t)
    c2w = w2c.inv()
    poses = c2w[0].inv() @ c2w  # view 0 = identity
    xyz = torch.tensor([[float(v) for v in l[1:4]] for l in lines("points3D.txt")])  # (N, 3) world
    d = w2c[0].transform(xyz).norm(dim=-1).median()  # distance from view 0, frame-invariant
    poses = Pose.from_Rt(poses.R, poses.t / d)

    fxfycxcy = torch.tensor([cams[r[8]] for r in rows]).to(offset)  # (V, 4)
    f = fxfycxcy[:, :2] * scale
    c = (fxfycxcy[:, 2:] - offset) * scale  # inverse of ZipSplat.forward's camera mapping
    return poses.to(offset.device), f, c


def select_views(enc, views):
    """ZipSplat.encode output with the tokens of some views only, cam_token keeps all views."""
    return {
        "layer_tokens": [t[:, views] for t in enc["layer_tokens"]],
        "geo_tokens": enc["geo_tokens"][:, views],
        "color_tokens": enc["color_tokens"][:, views],
        "cam_token": enc["cam_token"],
    }


def free_network(model):
    """Delete everything but the Gaussian head, BA needs the memory. Nothing can be decoded after."""
    for name in [n for n, _ in model.named_children() if n != "gaussian_head"]:
        delattr(model, name)
    gc.collect()
    torch.cuda.empty_cache()


def pose_drift(a, b):
    """Rotation (degrees) and camera center distance (scene units), each (V,), of poses (V,)."""
    return (a.inv() @ b).magnitude()


def drift_stats(name, a, b):
    """Mean and max pose_drift between poses a and b (V,), as metrics entries."""
    rot, trans = pose_drift(a, b)
    return {
        f"{name}_deg_mean": rot.mean().item(), f"{name}_deg_max": rot.max().item(),
        f"{name}_t_mean": trans.mean().item(), f"{name}_t_max": trans.max().item(),
    }


def moved_start(test, train_old, train_new):
    """Test poses (V_te,) carried along by the correction of the nearest train view.

    BA does not anchor view 0 and only moves the train poses (V_tr,), the test poses stay behind.
    """
    near = torch.cdist(test.t, train_old.t).argmin(1)  # (V_te,), nearest camera center
    return train_new[near] @ train_old[near].inv() @ test


def foreground_metrics(rgb, gt, m):
    """psnr_fg, ssim_fg, lpips_fg averaged over the views with foreground, as in main.

    Args: rgb, gt (V, 3, H, W) in [0, 1], mask m (V, 1, H, W).
    """
    ok = m.sum((1, 2, 3)) > 0
    if not ok.any():
        return {"psnr_fg": None, "ssim_fg": None, "lpips_fg": None}
    hat, gt, m = rgb.clamp(0, 1).float()[ok], gt.float()[ok], m[ok]
    n = m.sum((1, 2, 3))  # (V,)
    psnr = -10 * ((((hat - gt) ** 2) * m).sum((1, 2, 3)) / (3 * n)).log10()
    ssim_map = torch.stack([
        torch.from_numpy(structural_similarity(
            g.cpu().numpy(), h.cpu().numpy(), win_size=11, gaussian_weights=True,
            channel_axis=0, data_range=1.0, full=True,
        )[1])
        for g, h in zip(gt, hat)
    ]).to(m)  # (V, 3, H, W)
    ssim = (ssim_map * m).sum((1, 2, 3)) / (3 * n)
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(gt.device)
        lpips_map = torch.cat([lpips_fn(g[None], h[None], normalize=True) for g, h in zip(gt, hat)])
    lpips_fg = (lpips_map * m).sum((1, 2, 3)) / n
    return {"psnr_fg": psnr.mean().item(), "ssim_fg": ssim.mean().item(), "lpips_fg": lpips_fg.mean().item()}


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
    parser.add_argument(
        "--sparse", type=Path, default=None,
        help="COLMAP text model (cameras/images/points3D.txt): its poses and intrinsics are fed "
        "to the backbone as camera priors",
    )
    parser.add_argument(
        "--compression", type=float, default=1.0,
        help="compression r: fraction of the kept tokens that become scene tokens, (0, 1]",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="minimum foreground fraction (exclusive) of a patch to keep its token",
    )
    parser.add_argument("--alpha-weight", type=float, default=1.0, help="opacity loss weight of the BA")
    parser.add_argument(
        "--init-ba-rounds", type=int, default=0,
        help="start the BA from the poses of an unmasked TokenBA with this many PnP rounds, 0 = off",
    )
    parser.add_argument("--out", type=Path, default=None, help="output folder, default: outputs/masking_<time>")
    args = parser.parse_args()
    assert args.ba or not args.init_ba_rounds, "--init-ba-rounds needs --ba"
    assert 0 < args.compression <= 1, "--compression must be in (0, 1]"

    # One folder per run, next to this script.
    out = args.out or Path(__file__).parent / "outputs" / f"masking_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Masks first: a missing or mis-sized mask fails before ZipSplat loads.
    masks = load_masks(args.masks, args.images)
    images = [load_image(p) for p in args.images]

    # The wrapper loads the weights and crops the inputs, the network itself is wrapper.model.
    wrapper = ZipSplat(weights=args.weights).cuda().eval()
    model = wrapper.model

    # Same center crop and resize for images and masks.
    x, offset, scale = wrapper.prepare_inputs(images)
    x, mask = x[None], prepare_masks(wrapper, masks)  # (1, V, 3, H, W), (1, V, H, W)

    # Optional camera priors from COLMAP (c2w, view 0 = identity), at the model input resolution.
    priors = {}
    if args.sparse:
        V, H, W = x.shape[1], x.shape[-2], x.shape[-1]
        prior_poses, f, c = load_colmap_priors(args.sparse, [p.name for p in args.images], offset, scale)
        size = torch.tensor([W, H], dtype=f.dtype, device=f.device).expand(V, 2)
        prior_cams = Camera(torch.cat([size, f, c], -1))
        assert prior_poses.shape == (V,) and prior_cams.shape == (V,)
        assert torch.allclose(prior_poses[0].Rt, torch.eye(4, device=x.device), atol=1e-5)
        priors = {"poses": prior_poses[None], "cameras": prior_cams[None]}  # (1, V)

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
        enc = model.encode(x, **priors)
        if args.init_ba_rounds:  # before model.cluster is replaced and the network freed
            init_tokens = model.decode(select_views(enc, tr), x[:, tr])["scene_tokens"]  # (1, V_tr * T, D)
            assert init_tokens.shape[1] == len(tr) * enc["geo_tokens"].shape[2]
        enc_masked, keep = apply_token_masking(enc, mask_tok, model.patch_size, args.threshold)
        del enc  # no reference to the full encoding: it is freed before decode
        # k-means (compression < 1) has to seed from the foreground tokens only
        model.cluster = partial(masked_cluster, keep=keep, patch_size=model.patch_size)
        pred = model.decode(enc_masked, x, compression=args.compression)
    net_poses = pred["poses"]  # (1, V), before any BA
    ba_metrics = {}
    if args.ba:
        torch.cuda.synchronize()
        peak_net = torch.cuda.max_memory_allocated()
        free_network(model)
        torch.cuda.reset_peak_memory_stats()
        start = net_poses[:, tr]
        if args.init_ba_rounds:
            t1 = time.perf_counter()
            start = TokenBA(rounds=args.init_ba_rounds)(
                init_tokens, model.gaussian_head, pred["cameras"][:, tr], start, x[:, tr]
            )["poses"]
            del init_tokens
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            ba_metrics["time_init_s"] = time.perf_counter() - t1
            assert start.shape == (1, len(tr))
            ba_metrics.update(drift_stats("drift_init", net_poses[0, tr], start[0]))
        # batched (B == 1), outside autocast like ZipSplat.refine
        ba_out = MaskedTokenBA(mask[:, tr], alpha_weight=args.alpha_weight, rounds=0)(
            pred["scene_tokens"], model.gaussian_head, pred["cameras"][:, tr], start, x[:, tr]
        )
        peak_ba = torch.cuda.max_memory_allocated()
        ba_metrics["peak_mem_ba_gb"] = peak_ba / 1e9
        poses = pred["poses"].data_.clone()  # (1, V, D): refined train poses, network test poses
        poses[:, tr] = ba_out.pop("poses").data_
        pred = {**pred, **ba_out, "poses": type(pred["poses"])(poses)}  # refined gaussians, scene_tokens
    torch.cuda.synchronize()
    time_s = time.perf_counter() - t0  # encode -> masking -> decode (+ MaskedTokenBA if --ba)
    if args.ba:
        ba_metrics.update(drift_stats("drift_train", net_poses[0, tr], pred["poses"][0, tr]))
        ba_metrics.update(drift_stats("drift_view0", net_poses[0, :1], pred["poses"][0, :1]))
    if args.holdout:
        # Fair evaluation: every view is rendered with a pose aligned to the fixed scene
        # (joint_token_lr=0 keeps the scene tokens unchanged). With BA only the test views, BA
        # already refined the train poses; without BA all views. Not timed, it is evaluation.
        align = te if args.ba else idx
        aligned_poses = MaskedTokenBA(mask[:, align], rounds=0, joint_token_lr=0.0)(
            pred["scene_tokens"], model.gaussian_head, pred["cameras"][:, align], pred["poses"][:, align],
            x[:, align],
        )["poses"]
        poses = pred["poses"].data_.clone()
        poses[:, align] = aligned_poses.data_
        pred["poses"] = type(pred["poses"])(poses)
        if len(te):
            ba_metrics.update(drift_stats("align_test", net_poses[0, te], pred["poses"][0, te]))
        if args.ba and len(te):
            # Diagnostic: same alignment, but the test poses first follow the train pose correction.
            moved = moved_start(net_poses[0, te], net_poses[0, tr], pred["poses"][0, tr])
            moved = MaskedTokenBA(mask[:, te], rounds=0, joint_token_lr=0.0)(
                pred["scene_tokens"], model.gaussian_head, pred["cameras"][:, te], moved[None], x[:, te]
            )["poses"]
            with torch.no_grad():
                moved_rgb, _ = pred["gaussians"].render(pred["cameras"][:, te], moved)  # (1, V_te, 3, H, W)
    pred = remove_batch_dim(pred)

    # Checks: K = r * kept tokens scene tokens (all at r = 1), G Gaussians per scene token.
    N_fg = int(keep.sum())
    K = N_fg if args.compression >= 1.0 else max(1, int(N_fg * args.compression))
    G = model.gaussian_head.gaussians_per_token
    assert enc_masked["geo_tokens"].shape[2] == N_fg
    assert pred["scene_tokens"].shape[0] == K
    assert pred["gaussians"].shape[0] == K * G
    print(f"kept {N_fg} / {keep.numel()} tokens -> {K} scene tokens -> {pred['gaussians'].shape[0]} Gaussians")

    # Render the Gaussians back into the input views, next to the inputs.
    with torch.no_grad():
        rgb, infos = pred["gaussians"].render(pred["cameras"], pred["poses"])  # (V, 3, H, W)
    out.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([x[0], rgb.clamp(0, 1)], dim=0)  # inputs in row 1, renders in row 2
    save_image(grid, out / "inputs_vs_render.png", nrow=x.shape[1])
    save_image(mask[0, :, None], out / "masks.png", nrow=x.shape[1])
    pred["gaussians"].save_ply(out / "scene.ply")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9  # before the LPIPS network is loaded
    if args.ba:
        peak_mem_gb = max(peak_mem_gb, peak_net / 1e9)  # the peak was reset before the BA
        if len(te):
            moved_fg = foreground_metrics(moved_rgb[0], x[0, te], mask[0, te][:, None])
            ba_metrics.update({f"{k}_test_moved": v for k, v in moved_fg.items()})

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
        "scene_tokens": K,
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
        **ba_metrics,
        **({"test_views": te.tolist(), "pose_aligned_views": align.tolist(), **split} if args.holdout else {}),
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
