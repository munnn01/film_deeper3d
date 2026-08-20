"""Evaluate anchor and preprocessed clips with real H.264/H.265 codecs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import torch
from torch.nn import functional as F
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from preprocessing import FrozenVideoAnalyzer, StandardVideoCodec, build_preprocessor
from preprocessing.evaluation import build_evaluation_dataset, calculate_bd_rate
from preprocessing.utils import topk_correct, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-dir")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--val-ratio",
        type=float,
        help="override checkpoint validation ratio when val/ is absent",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="override checkpoint split seed when val/ is absent",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=("h264", "h265"),
        default=["h264", "h265"],
    )
    parser.add_argument("--qps", nargs="+", type=int, default=[30, 35, 40, 45])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--preset")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/real_codec")
    return parser.parse_args()


def format_bd_rate(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.2f}%"


def save_rate_accuracy_plot(
    path: Path,
    rows: list[dict[str, float | int | str]],
    codec: str,
    task_bd_rate: float | None,
    psnr_bd_rate: float | None,
) -> None:
    colors = {"anchor": "#E45756", "preprocessed": "#4C78A8"}
    labels = {"anchor": "Anchor", "preprocessed": "Video Swin preprocessor"}
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    codec_rows = [row for row in rows if row["codec"] == codec]
    for method in ("anchor", "preprocessed"):
        method_rows = [row for row in codec_rows if row["method"] == method]
        qp_rows = sorted(method_rows, key=lambda row: int(row["qp"]))
        rate_rows = sorted(method_rows, key=lambda row: float(row["bpp"]))
        color = colors[method]
        axes[0].plot(
            [row["qp"] for row in qp_rows],
            [row["bpp"] for row in qp_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=labels[method],
        )
        axes[1].plot(
            [row["bpp"] for row in rate_rows],
            [row["top1_percent"] for row in rate_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=labels[method],
        )
        axes[2].plot(
            [row["bpp"] for row in rate_rows],
            [row["psnr_db"] for row in rate_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=labels[method],
        )
        for row in rate_rows:
            axes[1].annotate(
                f"QP {int(row['qp'])}",
                (float(row["bpp"]), float(row["top1_percent"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axes[0].set(xlabel="QP", ylabel="Bitrate (BPP)", title="QP - BPP")
    axes[1].set(
        xlabel="Bitrate (BPP)",
        ylabel="Top-1 accuracy (%)",
        title="Task rate-accuracy",
    )
    axes[2].set(
        xlabel="Bitrate (BPP)",
        ylabel="PSNR (dB)",
        title="Classical rate-distortion",
    )
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(
        f"{codec.upper()} | Task BD-rate: {format_bd_rate(task_bd_rate)} | "
        f"PSNR BD-rate: {format_bd_rate(psnr_bd_rate)}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def real_codec_roundtrip(
    clip: torch.Tensor,
    codec: str,
    qp: int,
    fps: float,
    *,
    preset: str = "medium",
    ffmpeg: str = "ffmpeg",
) -> tuple[torch.Tensor, float]:
    module = StandardVideoCodec(
        codec, qp, fps=fps, preset=preset, ffmpeg=ffmpeg
    )
    reconstruction, bpp = module(clip.unsqueeze(0))
    return reconstruction[0], float(bpp[0])


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    codec_fps = args.fps if args.fps is not None else float(saved_args.get("codec_fps", 30.0))
    codec_preset = args.preset or saved_args.get("codec_preset", "medium")
    analyzer_name = saved_args.get("analyzer", "r3d_18")
    preprocessor_kind = saved_args.get("preprocessor", "cnn")
    analyzer = FrozenVideoAnalyzer(analyzer_name).to(device).eval()
    preprocessor = build_preprocessor(
        preprocessor_kind,
        temporal_frames=int(saved_args.get("temporal_frames", 8)),
        patch_size=int(saved_args.get("vit_patch_size", 8)),
        embed_dim=int(saved_args.get("vit_embed_dim", 96)),
        depth=int(saved_args.get("vit_depth", 4)),
        num_heads=int(saved_args.get("vit_heads", 4)),
        swin_patch_size=int(saved_args.get("swin_patch_size", 4)),
        swin_embed_dim=int(saved_args.get("swin_embed_dim", 48)),
        swin_depth=int(saved_args.get("swin_depth", 4)),
        swin_num_heads=int(saved_args.get("swin_heads", 4)),
        swin_window_size=(
            int(saved_args.get("swin_window_temporal", 4)),
            int(saved_args.get("swin_window_spatial", 8)),
            int(saved_args.get("swin_window_spatial", 8)),
        ),
        max_residual=float(saved_args.get("max_residual", 0.25)),
    ).to(device).eval()
    preprocessor.load_state_dict(checkpoint["preprocessor"])

    dataset = build_evaluation_dataset(
        data_root=args.data_root,
        test_dir=args.test_dir,
        split=args.split,
        categories=analyzer.categories,
        frames=args.frames,
        stride=args.frame_stride,
        size=args.frame_size,
        limit=args.limit,
        val_ratio=args.val_ratio,
        seed=args.seed,
        saved_args=saved_args,
    )
    totals: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"videos": 0, "bpp": 0.0, "mse": 0.0, "top1": 0, "top5": 0}
    )

    for clip, label in tqdm(dataset, desc="real codec evaluation"):
        source = clip.to(device).unsqueeze(0)
        with torch.no_grad():
            proposed = preprocessor(source)[0].cpu()
        for codec in args.codecs:
            for qp in args.qps:
                for method, input_clip in (("anchor", clip), ("preprocessed", proposed)):
                    decoded, bpp = real_codec_roundtrip(
                        input_clip,
                        codec,
                        qp,
                        codec_fps,
                        preset=codec_preset,
                        ffmpeg=args.ffmpeg,
                    )
                    decoded_device = decoded.to(device).unsqueeze(0)
                    label_tensor = torch.tensor([label], device=device)
                    with torch.no_grad():
                        logits = analyzer(decoded_device)
                    key = (codec, qp, method)
                    row = totals[key]
                    row["videos"] += 1
                    row["bpp"] += bpp
                    row["mse"] += float(F.mse_loss(decoded_device, source))
                    row["top1"] += topk_correct(logits, label_tensor, 1)
                    row["top5"] += topk_correct(logits, label_tensor, 5)

    rows = []
    for (codec, qp, method), values in sorted(totals.items()):
        count = int(values["videos"])
        mse = values["mse"] / count
        rows.append(
            {
                "codec": codec,
                "qp": qp,
                "method": method,
                "videos": count,
                "bpp": values["bpp"] / count,
                "mse": mse,
                "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
                "top1": values["top1"] / count,
                "top1_percent": 100.0 * values["top1"] / count,
                "top5": values["top5"] / count,
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", rows)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output / 'metrics.csv'}")

    bd_rates = {}
    for codec in args.codecs:
        codec_rows = [row for row in rows if row["codec"] == codec]
        task_bd_rate = calculate_bd_rate(codec_rows, "top1_percent")
        psnr_bd_rate = calculate_bd_rate(codec_rows, "psnr_db")
        bd_rates[codec] = {
            "task_bd_rate_percent": task_bd_rate,
            "psnr_bd_rate_percent": psnr_bd_rate,
        }
        plot_path = output / f"{codec}_top1_bpp_bd_rate.png"
        save_rate_accuracy_plot(
            plot_path, rows, codec, task_bd_rate, psnr_bd_rate
        )
        print(
            f"[{codec}] task BD-rate={format_bd_rate(task_bd_rate)} "
            f"PSNR BD-rate={format_bd_rate(psnr_bd_rate)}"
        )
        print(f"wrote {plot_path}")
    write_json(output / "bd_rate.json", bd_rates)
    print(f"wrote {output / 'bd_rate.json'}")


if __name__ == "__main__":
    main()
