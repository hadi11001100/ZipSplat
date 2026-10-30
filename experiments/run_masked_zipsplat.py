"""
ZipSplat with background-masked tokens: images -> RMBG masks -> encode -> token masking -> decode.

The masking itself lives in experiments/masking.py, this script only wires it into ZipSplat and
saves the results to experiments/outputs/masking_<date>_<time>/.

Usage:
    python -m experiments.run_masked_zipsplat assets/examples/office/*.png
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torchvision.utils import save_image

from experiments.masking import apply_token_masking, get_background_mask, prepare_masks
from zipsplat import ZipSplat, load_image
from zipsplat.utils import remove_batch_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default="zipsplat")
    args = parser.parse_args()

    # One folder per run, next to this script.
    out = Path(__file__).parent / "outputs" / f"masking_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Masks first: RMBG is freed before ZipSplat goes to the GPU.
    images = [load_image(p) for p in args.images]
    masks = get_background_mask(images)

    # The wrapper loads the weights and crops the inputs, the network itself is wrapper.model.
    wrapper = ZipSplat(weights=args.weights).cuda().eval()
    model = wrapper.model

    # Same center crop and resize for images and masks.
    x, _, _ = wrapper.prepare_inputs(images)
    x, mask = x[None], prepare_masks(wrapper, masks)  # (1, V, 3, H, W), (1, V, H, W)

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
    out.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([x[0], rgb.clamp(0, 1)], dim=0)  # inputs in row 1, renders in row 2
    save_image(grid, out / "inputs_vs_render.png", nrow=x.shape[1])
    save_image(mask[0, :, None], out / "masks.png", nrow=x.shape[1])
    pred["gaussians"].save_ply(out / "scene.ply")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
