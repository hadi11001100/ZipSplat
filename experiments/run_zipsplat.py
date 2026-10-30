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
from datetime import datetime
from pathlib import Path

import torch
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
    pred = wrapper(images, compression=1.0, ba=args.ba)
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
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
