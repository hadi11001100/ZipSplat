"""
Background masking of the DA3 tokens, between ZipSplat's encode and decode.

Module only, the pipeline lives in experiments/run_masked_zipsplat.py. The masked tokens of all
views are packed into one pseudo-view (B, 1, N_fg, D). The fusion blocks carry no positional
encoding, so decode accepts any token count.

The foreground comes from hand-made YOLO polygon labels (CVAT), a token is kept when the
average-pooled mask over its patch exceeds a threshold.

Limits for now:
    - compression=1.0 only: complexity_kmeans seeds from all V * T patches of the images.
    - TokenBA only as MaskedTokenBA with rounds=0: the PnP camera update expects all V * T tokens.
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from zipsplat.token_ba import TokenBA


def load_masks(label_dir, image_paths):
    """Binary foreground masks [(1, h, w), ...] of the raw views, from YOLO segmentation labels.

    Args:
        label_dir: Folder with one YOLO label file per view, named like the image (<stem>.txt).
            Each line is "cls x1 y1 x2 y2 ...", a polygon in coordinates normalized to [0, 1].
        image_paths: Image path per view, gives the mask size.

    All polygons of a view are filled and merged, regardless of their class.
    """
    masks = []
    for path in image_paths:
        label_path = label_dir / f"{path.stem}.txt"
        assert label_path.exists(), f"no label for {path.name}: {label_path}"
        polygons = [line.split()[1:] for line in label_path.read_text().splitlines() if line.strip()]
        assert polygons, f"no polygon in {label_path}"
        w, h = Image.open(path).size
        canvas = Image.new("L", (w, h), 0)
        for coords in polygons:
            assert len(coords) % 2 == 0 and len(coords) >= 6, f"bad polygon in {label_path}"
            xy = [(float(x) * w, float(y) * h) for x, y in zip(coords[::2], coords[1::2])]
            ImageDraw.Draw(canvas).polygon(xy, fill=255)
        masks.append((to_tensor(canvas) > 0.5).float())  # (1, h, w)
    return masks


def prepare_masks(wrapper, masks, threshold=0.5):
    """Binary masks (1, V, H, W) with the same center crop and resize as the model images.

    Args:
        wrapper: ZipSplat wrapper, its prepare_inputs crops and resizes the images.
        masks: Foreground masks [(1, h, w), ...] from load_masks.
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


class MaskedTokenBA(TokenBA):
    """TokenBA on masked tokens: RGB loss on the foreground, opacity loss on the whole image.

    The background has no Gaussians, so the RGB L1 only counts foreground pixels. The opacity
    term pulls the rendered alpha to the mask (1 inside, 0 outside), so foreground Gaussians
    cannot grow into the background (no halo / floaters at the mask edge).
    Run with rounds=0: the PnP camera update expects the full V * T patch grid.

    Args:
        mask: Foreground mask (1, V, H, W) in {0, 1}, same crop and size as the images.
        alpha_weight: Weight of the opacity term.
        **kwargs: TokenBA settings.
    """

    def __init__(self, mask, alpha_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.mask = mask[:, :, None].float()  # (1, V, 1, H, W)
        self.alpha_weight = alpha_weight

    def _step(self, optimizer, head, scene_tokens, camera, pose, image):
        optimizer.zero_grad(set_to_none=True)
        gaussians = head(scene_tokens, half_mlp=self.half_decoder)
        rgb, infos = gaussians.render(camera, pose)  # (1, V, 3, H, W), alphas (1, V, 1, H, W)
        m = self.mask
        loss_rgb = ((rgb - image).abs() * m).sum() / (3 * m.sum()).clamp(min=1)
        loss_alpha = (infos["alphas"] - m).abs().mean()
        loss = loss_rgb + self.alpha_weight * loss_alpha
        loss.backward()
        optimizer.step()
        return loss.detach()
