"""
Draw one box around the object per image, by hand. The boxes later limit the RMBG masks.

Shows the images one after another. Drag a rectangle with the left mouse button (it can be
moved and resized afterwards), Enter saves it and shows the next image, Esc clears it. Closing
the window stops, the boxes so far are kept.

The boxes are saved after every image to a JSON file {image name: [x0, y0, x1, y1]} in pixels
of the original image, x1 and y1 exclusive. Images that already have a box are skipped, so a
stopped run continues where it left off.

Usage:
    python -m experiments.draw_boxes experiments/rig/ImageTest2/*.jpg --out experiments/boxes.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image


def ask_box(path, title=""):
    """Box [x0, y0, x1, y1] the user draws on the image at path, None if the window is closed."""
    image = Image.open(path)
    w, h = image.size
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)
    ax.set_title(f"{title}{path.name}   drag: box   Enter: save   Esc: clear   close: stop")
    ax.set_axis_off()
    fig.tight_layout()

    state = {"drawn": False, "box": None}

    def on_select(press, release):
        state["drawn"] = True

    # interactive: the box keeps handles to move and resize it before Enter
    selector = RectangleSelector(ax, on_select, button=[1], interactive=True, useblit=True)

    def on_key(event):
        if event.key == "escape":  # the selector clears itself on Esc
            state["drawn"] = False
        elif event.key == "enter" and state["drawn"]:
            x0, x1, y0, y1 = selector.extents
            box = [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)]
            box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
            if box[2] > box[0] and box[3] > box[1]:
                state["box"] = box
                plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()  # blocks until the window is closed
    return state["box"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "boxes.json")
    args = parser.parse_args()

    plt.switch_backend("TkAgg")
    boxes = json.loads(args.out.read_text()) if args.out.exists() else {}
    todo = [p for p in args.images if p.name not in boxes]
    print(f"{len(args.images) - len(todo)} images already have a box, {len(todo)} to go")

    for i, path in enumerate(todo):
        box = ask_box(path, title=f"[{i + 1}/{len(todo)}]  ")
        if box is None:
            print(f"stopped at {path.name}")
            break
        boxes[path.name] = box
        args.out.write_text(json.dumps(boxes, indent=2))  # after every image: nothing gets lost
    print(f"{len(boxes)} boxes in {args.out}")


if __name__ == "__main__":
    main()
