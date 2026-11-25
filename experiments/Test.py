"""
Runs the ZipSplat experiments over all sensible parameter combinations, one subprocess per case.

Only orchestration: each case calls experiments/run_zipsplat.py (unmasked) or
experiments/run_masked_zipsplat.py (masked), so an out-of-memory error ends one case only.
Results: experiments/outputs/test_<date>_<time>/<case>/ and results.csv (updated after each case).

The test views are held out, compare psnr_fg_test. A large gap between psnr_fg_test_moved and
psnr_fg_test means that the test camera did not follow the pose drift of the BA.

Usage:
    source .venv/bin/activate
    python -m experiments.Test experiments/rig/ImageTest2/*.jpg --masks experiments/rig/labels --dry-run
    python -m experiments.Test experiments/rig/ImageTest2/*.jpg --masks experiments/rig/labels
    python -m experiments.Test experiments/rig/ImageTest2/*.jpg --masks experiments/rig/labels \
        --filter plain_ba --compression 1.0 0.25 --sparse experiments/sparse/0
"""

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PARAMS = ["masks", "ba", "rounds", "threshold", "alpha_weight", "init_ba_rounds", "compression", "sparse"]
METRICS = [
    "psnr_fg_test", "psnr_fg_test_moved", "psnr_fg_train", "ssim_fg_test", "ssim_fg_test_moved",
    "ssim_fg_train", "lpips_fg_test", "lpips_fg_test_moved", "lpips_fg_train", "alpha_bg",
    "drift_train_deg_mean", "drift_train_deg_max", "drift_train_t_mean", "drift_train_t_max",
    "drift_view0_deg_max", "drift_view0_t_max", "drift_init_deg_mean", "drift_init_t_mean",
    "align_test_deg_mean", "align_test_deg_max", "align_test_t_mean", "align_test_t_max",
    "tokens_kept", "num_gaussians", "time_s", "time_init_s", "peak_mem_gb", "peak_mem_ba_gb",
]


def build_cases(thresholds, compressions, sparse):
    """All sensible combinations, {name: params}. Parameters without effect are left out."""
    cases = {}
    for r, prior in itertools.product(compressions, [None, sparse] if sparse else [None]):
        suffix = f"_c{r:g}" + ("_colmap" if prior else "")
        common = {"compression": r, "sparse": prior}

        def add(name, **params):
            assert name + suffix not in cases, f"duplicate case {name + suffix}"
            cases[name + suffix] = {**params, **common}

        add("plain", masks=False, ba=False)
        for rounds in (0, 4):
            add(f"plain_ba_r{rounds}", masks=False, ba=True, rounds=rounds)
        for t in thresholds:
            add(f"masked_t{t:g}", masks=True, ba=False, threshold=t)
            # rounds=0 always: PnP needs the full patch grid, so it only gives the start poses.
            for alpha, init in itertools.product((0.0, 1.0), (0, 4)):
                add(
                    f"masked_t{t:g}_ba_a{alpha:g}_init{init}", masks=True, ba=True, rounds=0,
                    threshold=t, alpha_weight=alpha, init_ba_rounds=init,
                )
    return cases


def build_command(params, out, config):
    """Command line of one case."""
    script = "run_masked_zipsplat" if params["masks"] else "run_zipsplat"
    cmd = [sys.executable, "-m", f"experiments.{script}", *config["images"]]
    cmd += ["--masks", config["masks"], "--holdout", str(config["holdout"]), "--out", str(out)]
    cmd += ["--compression", str(params["compression"])]
    if params["sparse"]:
        cmd += ["--sparse", params["sparse"]]
    if params["ba"]:
        cmd += ["--ba"]
    if params["masks"]:
        cmd += ["--threshold", str(params["threshold"])]
        if params["ba"]:
            cmd += ["--alpha-weight", str(params["alpha_weight"])]
            cmd += ["--init-ba-rounds", str(params["init_ba_rounds"])]
    elif params["ba"]:
        cmd += ["--rounds", str(params["rounds"])]
    return cmd


def run_case(params, out, config):
    """Run one case unless it is done already, returns its status: ok, oom or failed."""
    if (out / "metrics.json").exists():
        return "ok"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "log.txt", "w") as log:
        code = subprocess.run(
            build_command(params, out, config), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    if code == 0 and (out / "metrics.json").exists():
        return "ok"
    log = (out / "log.txt").read_text(errors="replace")
    return "oom" if "OutOfMemoryError" in log or "out of memory" in log else "failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--masks", type=Path, required=True, help="YOLO polygon labels, one <stem>.txt per image")
    parser.add_argument("--holdout", type=int, default=8, help="every K-th view is a test view")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--compression", type=float, nargs="+", default=[1.0])
    parser.add_argument("--sparse", type=Path, default=None, help="also run every case with these COLMAP priors")
    parser.add_argument("--filter", default="", help="only cases whose name contains this text")
    parser.add_argument("--dry-run", action="store_true", help="only list the cases")
    parser.add_argument("--resume", type=Path, default=None, help="continue this test folder, skips finished cases")
    args = parser.parse_args()
    assert args.holdout >= 2, "--holdout must be >= 2, the cases are compared on the test views"
    assert len(args.images) >= args.holdout, "no test view: fewer images than --holdout"

    cases = build_cases(args.thresholds, args.compression, str(args.sparse.resolve()) if args.sparse else None)
    cases = {n: p for n, p in cases.items() if args.filter in n}
    if args.dry_run:
        print("\n".join(cases))
        print(f"{len(cases)} cases")
        return 0
    assert cases, f"no case contains {args.filter!r}"

    # Everything that has to be equal for the cases of one folder to be comparable.
    config = {
        "images": [str(p.resolve()) for p in args.images],
        "masks": str(args.masks.resolve()),
        "holdout": args.holdout,
    }
    out = args.resume or ROOT / "experiments" / "outputs" / f"test_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out = out.resolve()
    if args.resume:
        assert json.loads((out / "config.json").read_text()) == config, "--resume with different inputs"
    else:
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps(config, indent=2))

    rows = []
    for i, (name, params) in enumerate(cases.items()):
        print(f"[{i + 1}/{len(cases)}] {name} ...", end=" ", flush=True)
        status = run_case(params, out / name, config)
        metrics = json.loads((out / name / "metrics.json").read_text()) if status == "ok" else {}
        rows.append({
            "case": name, "status": status,
            **{k: params.get(k) for k in PARAMS}, **{k: metrics.get(k) for k in METRICS},
        })
        print(status if status == "ok" else f"{status}, see {out / name / 'log.txt'}", flush=True)
        with open(out / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "status", *PARAMS, *METRICS])
            writer.writeheader()
            writer.writerows(rows)

    rows.sort(key=lambda r: r["psnr_fg_test"] if r["psnr_fg_test"] is not None else float("-inf"), reverse=True)
    show = ["psnr_fg_test", "psnr_fg_test_moved", "psnr_fg_train", "drift_train_deg_max", "num_gaussians", "time_s"]
    print(f"\n{'case':<40} {'status':<7}" + "".join(f"{k:>21}" for k in show))
    for r in rows:
        cells = "".join(f"{'-' if r[k] is None else format(r[k], '.2f'):>21}" for k in show)
        print(f"{r['case']:<40} {r['status']:<7}" + cells)
    print(f"saved to {out / 'results.csv'}")
    return int(any(r["status"] != "ok" for r in rows))


if __name__ == "__main__":
    sys.exit(main())
