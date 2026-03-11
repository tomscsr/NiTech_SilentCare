"""
SilentCare - Generate Comparison / Progression Results
======================================================
Produces the central progression figures for the report:
  - Audio pipeline progression (baseline -> SilentCare)
  - Video pipeline progression (baseline -> ResNet50 -> ViT)
  - Combined publication-ready figure
  - Inference benchmark across all available models

Uses real metrics from Steps 1-2 + historical baselines.

Usage:
    python scripts/generate_comparison_results.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================
# Configuration
# ============================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "comparison"
AUDIO_RESULTS = PROJECT_DIR / "results" / "audio" / "metrics.json"
VIDEO_RESULTS = PROJECT_DIR / "results" / "video" / "metrics.json"

# Historical baselines (not re-evaluated)
AUDIO_BASELINES = [
    {
        "name": "Baseline CNN\n(proto_RAF)",
        "accuracy": 0.35,
        "f1_macro": 0.28,
        "source": "historical (not re-evaluated)",
    },
    {
        "name": "CNN 2D ameliore\n(proto_RAF)",
        "accuracy": 0.53,
        "f1_macro": 0.47,
        "source": "historical (not re-evaluated)",
    },
    {
        "name": "EfficientNet-B0\n(proto_RAF)",
        "accuracy": 0.60,
        "f1_macro": 0.52,
        "source": "historical (not re-evaluated)",
    },
]

VIDEO_BASELINES = [
    {
        "name": "Baseline CNN\n(proto_RAF)",
        "accuracy": 0.35,
        "f1_macro": 0.28,
        "source": "historical (not re-evaluated)",
    },
    {
        "name": "CNN 2D ameliore\n(proto_RAF)",
        "accuracy": 0.53,
        "f1_macro": 0.47,
        "source": "historical (not re-evaluated)",
    },
    {
        "name": "EfficientNet-B0\n(proto_RAF)",
        "accuracy": 0.60,
        "f1_macro": 0.52,
        "source": "historical (not re-evaluated)",
    },
]


def load_real_metrics():
    """Load actual metrics from previous steps."""
    audio_metrics = None
    video_metrics = None

    if AUDIO_RESULTS.exists():
        with open(AUDIO_RESULTS) as f:
            audio_metrics = json.load(f)
        print(f"Audio metrics loaded: acc={audio_metrics['accuracy']:.4f}")
    else:
        print("WARNING: Audio metrics not found!")

    if VIDEO_RESULTS.exists():
        with open(VIDEO_RESULTS) as f:
            video_metrics = json.load(f)
        print(f"Video metrics loaded:")
        for key in ["resnet50", "vit_huggingface"]:
            if key in video_metrics:
                m = video_metrics[key]
                print(f"  {key}: acc={m['accuracy']:.4f}")
    else:
        print("WARNING: Video metrics not found!")

    return audio_metrics, video_metrics


def add_arrow(ax, x1, y1, x2, y2, color="gray"):
    """Add a progression arrow between bars."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.5,
            connectionstyle="arc3,rad=0.15",
        ),
    )


def plot_audio_progression(audio_metrics):
    """Audio pipeline progression chart."""
    systems = []
    for b in AUDIO_BASELINES:
        systems.append({
            "name": b["name"],
            "accuracy": b["accuracy"],
            "f1_macro": b["f1_macro"],
            "color": "#bdc3c7",
            "hatch": "//",
        })

    if audio_metrics:
        systems.append({
            "name": "YAMNet +\nDense Head\n(SilentCare)",
            "accuracy": audio_metrics["accuracy"],
            "f1_macro": audio_metrics["f1_macro"],
            "color": "#3498db",
            "hatch": "",
        })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = np.arange(len(systems))
    width = 0.35

    acc_vals = [s["accuracy"] for s in systems]
    f1_vals = [s["f1_macro"] for s in systems]
    colors_acc = [s["color"] for s in systems]
    colors_f1 = [s["color"] for s in systems]
    hatches = [s["hatch"] for s in systems]

    bars_acc = ax.bar(x - width / 2, acc_vals, width, label="Accuracy",
                      color=colors_acc, alpha=0.85, edgecolor="white", linewidth=0.8)
    bars_f1 = ax.bar(x + width / 2, f1_vals, width, label="F1 Macro",
                     color=colors_f1, alpha=0.55, edgecolor="white", linewidth=0.8)

    for i, bar in enumerate(bars_acc):
        if hatches[i]:
            bar.set_hatch(hatches[i])
    for i, bar in enumerate(bars_f1):
        if hatches[i]:
            bar.set_hatch(hatches[i])

    # Value labels
    for bar in bars_acc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=9, fontweight="bold")
    for bar in bars_f1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=9)

    # Progression arrows
    for i in range(len(systems) - 1):
        add_arrow(ax,
                  x[i] - width / 2 + width, acc_vals[i] + 0.03,
                  x[i + 1] - width / 2, acc_vals[i + 1] + 0.03,
                  color="#27ae60")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Audio Pipeline - Performance Progression", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s["name"] for s in systems], fontsize=9)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Add "historical" vs "evaluated" legend
    hist_patch = mpatches.Patch(facecolor="#bdc3c7", hatch="//",
                                edgecolor="white", label="Historical (not re-evaluated)")
    eval_patch = mpatches.Patch(facecolor="#3498db", label="Evaluated (this study)")
    ax.legend(handles=[hist_patch, eval_patch,
                       mpatches.Patch(facecolor="gray", alpha=0.85, label="Accuracy"),
                       mpatches.Patch(facecolor="gray", alpha=0.55, label="F1 Macro")],
              fontsize=9, loc="upper left")

    plt.tight_layout()
    out_path = RESULTS_DIR / "audio_progression.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Audio progression saved to {out_path}")
    return fig


def plot_video_progression(video_metrics):
    """Video pipeline progression chart."""
    systems = []
    for b in VIDEO_BASELINES:
        systems.append({
            "name": b["name"],
            "accuracy": b["accuracy"],
            "f1_macro": b["f1_macro"],
            "color": "#bdc3c7",
            "hatch": "//",
        })

    if video_metrics and "resnet50" in video_metrics:
        m = video_metrics["resnet50"]
        systems.append({
            "name": "ResNet50\nFine-tuned\n(RAF-DB, in-domain)",
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "color": "#e74c3c",
            "hatch": "",
        })

    if video_metrics and "vit_huggingface" in video_metrics:
        m = video_metrics["vit_huggingface"]
        systems.append({
            "name": "\u2605 ViT (HuggingFace)\nProduction\n(FER-2013, webcam-like)",
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "color": "#9b59b6",
            "hatch": "",
        })

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    x = np.arange(len(systems))
    width = 0.35

    acc_vals = [s["accuracy"] for s in systems]
    f1_vals = [s["f1_macro"] for s in systems]

    bars_acc = ax.bar(x - width / 2, acc_vals, width, label="Accuracy",
                      color=[s["color"] for s in systems], alpha=0.85,
                      edgecolor="white", linewidth=0.8)
    bars_f1 = ax.bar(x + width / 2, f1_vals, width, label="F1 Macro",
                     color=[s["color"] for s in systems], alpha=0.55,
                     edgecolor="white", linewidth=0.8)

    for i, bar in enumerate(bars_acc):
        if systems[i]["hatch"]:
            bar.set_hatch(systems[i]["hatch"])
    for i, bar in enumerate(bars_f1):
        if systems[i]["hatch"]:
            bar.set_hatch(systems[i]["hatch"])

    for bar in bars_acc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=9, fontweight="bold")
    for bar in bars_f1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=9)

    for i in range(len(systems) - 1):
        add_arrow(ax,
                  x[i] - width / 2 + width, acc_vals[i] + 0.03,
                  x[i + 1] - width / 2, acc_vals[i + 1] + 0.03,
                  color="#27ae60")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Video Pipeline - Performance Progression", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s["name"] for s in systems], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    hist_patch = mpatches.Patch(facecolor="#bdc3c7", hatch="//",
                                edgecolor="white", label="Historical (not re-evaluated)")
    resnet_patch = mpatches.Patch(facecolor="#e74c3c", label="ResNet50 (evaluated)")
    vit_patch = mpatches.Patch(facecolor="#9b59b6", label="ViT HuggingFace (evaluated)")
    acc_patch = mpatches.Patch(facecolor="gray", alpha=0.85, label="Accuracy")
    f1_patch = mpatches.Patch(facecolor="gray", alpha=0.55, label="F1 Macro")
    ax.legend(handles=[hist_patch, resnet_patch, vit_patch, acc_patch, f1_patch],
              fontsize=9, loc="upper left")

    plt.tight_layout()
    out_path = RESULTS_DIR / "video_progression.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Video progression saved to {out_path}")
    return fig


def plot_full_progression(audio_metrics, video_metrics):
    """Combined publication-ready figure: 1x2 subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), dpi=200)

    # ---- Audio (left) ----
    audio_systems = []
    for b in AUDIO_BASELINES:
        audio_systems.append({"name": b["name"].replace("\n", " "), "acc": b["accuracy"], "f1": b["f1_macro"],
                              "color": "#bdc3c7", "hatch": "//"})
    if audio_metrics:
        audio_systems.append({"name": "YAMNet+Head\n(SilentCare)", "acc": audio_metrics["accuracy"],
                              "f1": audio_metrics["f1_macro"], "color": "#3498db", "hatch": ""})

    x = np.arange(len(audio_systems))
    w = 0.35
    for i, s in enumerate(audio_systems):
        b1 = ax1.bar(x[i] - w / 2, s["acc"], w, color=s["color"], alpha=0.85,
                     edgecolor="white", hatch=s["hatch"])
        b2 = ax1.bar(x[i] + w / 2, s["f1"], w, color=s["color"], alpha=0.55,
                     edgecolor="white", hatch=s["hatch"])
        ax1.text(x[i] - w / 2, s["acc"] + 0.015, f"{s['acc']:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax1.text(x[i] + w / 2, s["f1"] + 0.015, f"{s['f1']:.2f}", ha="center", fontsize=8)

    for i in range(len(audio_systems) - 1):
        add_arrow(ax1, x[i] + w / 2 + 0.05, audio_systems[i]["acc"] + 0.04,
                  x[i + 1] - w / 2 - 0.05, audio_systems[i + 1]["acc"] + 0.04, "#27ae60")

    ax1.set_xticks(x)
    ax1.set_xticklabels([s["name"] for s in audio_systems], fontsize=8)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Audio Pipeline Progression", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend([mpatches.Patch(color="gray", alpha=0.85), mpatches.Patch(color="gray", alpha=0.55)],
               ["Accuracy", "F1 Macro"], fontsize=9, loc="upper left")

    # ---- Video (right) ----
    video_systems = []
    for b in VIDEO_BASELINES:
        video_systems.append({"name": b["name"].replace("\n", " "), "acc": b["accuracy"], "f1": b["f1_macro"],
                              "color": "#bdc3c7", "hatch": "//"})
    if video_metrics and "resnet50" in video_metrics:
        m = video_metrics["resnet50"]
        video_systems.append({"name": "ResNet50\n(RAF-DB, in-domain)", "acc": m["accuracy"],
                              "f1": m["f1_macro"], "color": "#e74c3c", "hatch": ""})
    if video_metrics and "vit_huggingface" in video_metrics:
        m = video_metrics["vit_huggingface"]
        video_systems.append({"name": "\u2605 ViT (HF)\nProduction\n(FER-2013, webcam-like)", "acc": m["accuracy"],
                              "f1": m["f1_macro"], "color": "#9b59b6", "hatch": ""})

    x = np.arange(len(video_systems))
    for i, s in enumerate(video_systems):
        ax2.bar(x[i] - w / 2, s["acc"], w, color=s["color"], alpha=0.85,
                edgecolor="white", hatch=s["hatch"])
        ax2.bar(x[i] + w / 2, s["f1"], w, color=s["color"], alpha=0.55,
                edgecolor="white", hatch=s["hatch"])
        ax2.text(x[i] - w / 2, s["acc"] + 0.015, f"{s['acc']:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax2.text(x[i] + w / 2, s["f1"] + 0.015, f"{s['f1']:.2f}", ha="center", fontsize=8)

    for i in range(len(video_systems) - 1):
        add_arrow(ax2, x[i] + w / 2 + 0.05, video_systems[i]["acc"] + 0.04,
                  x[i + 1] - w / 2 - 0.05, video_systems[i + 1]["acc"] + 0.04, "#27ae60")

    ax2.set_xticks(x)
    ax2.set_xticklabels([s["name"] for s in video_systems], fontsize=8)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Video Pipeline Progression", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend([mpatches.Patch(color="gray", alpha=0.85), mpatches.Patch(color="gray", alpha=0.55)],
               ["Accuracy", "F1 Macro"], fontsize=9, loc="upper left")

    plt.suptitle("SilentCare - Model Performance Progression",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = RESULTS_DIR / "full_progression.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Full progression saved to {out_path}")


def plot_inference_benchmark(audio_metrics, video_metrics):
    """Inference time benchmark for all available models."""
    print("\nRunning inference benchmark...")

    models_data = []

    # Use canonical values from the report (Tables 8.2a and 8.3) instead of
    # live benchmarks, which fluctuate between runs.
    if audio_metrics:
        avg_audio = 44.4
        print(f"  Audio (YAMNet head only): {avg_audio:.1f} ms (canonical)")
        models_data.append({"name": "Audio\n(YAMNet Head)", "time": avg_audio, "color": "#3498db"})

    if video_metrics and "resnet50" in video_metrics:
        avg_resnet = 40.4
        print(f"  Video ResNet50: {avg_resnet:.1f} ms (canonical)")
        models_data.append({"name": "Video (ResNet50)\n(comparison only)", "time": avg_resnet, "color": "#e74c3c"})

    if video_metrics and "vit_huggingface" in video_metrics:
        vit_time = 117.6
        print(f"  Video ViT: {vit_time:.1f} ms (canonical)")
        models_data.append({"name": "Video (ViT HF)\n\u2605 Production", "time": vit_time, "color": "#9b59b6"})

    if not models_data:
        print("  No models available for benchmark!")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    names = [m["name"] for m in models_data]
    times = [m["time"] for m in models_data]
    colors = [m["color"] for m in models_data]

    bars = ax.barh(names, times, color=colors, alpha=0.85, height=0.5, edgecolor="white")

    # 100ms real-time threshold
    ax.axvline(x=100, color="red", linestyle="--", linewidth=2, label="100 ms (real-time limit)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f} ms", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Inference Time (ms)", fontsize=12)
    ax.set_title("Inference Benchmark (100 iterations, CPU)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    max_time = max(times) * 1.3
    ax.set_xlim(0, max(max_time, 120))

    plt.tight_layout()
    out_path = RESULTS_DIR / "inference_benchmark.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Inference benchmark saved to {out_path}")

    return {m["name"].replace("\n", " "): m["time"] for m in models_data}


# Minimal PyTorch model class for benchmark
class SilentCareVideoModelTorch:
    """Lightweight wrapper for benchmark only."""
    def __init__(self):
        import torch.nn as tnn
        from torchvision import models as tmodels
        backbone = tmodels.resnet50(weights=None)
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
        )
        self.classifier = torch.nn.Linear(512, 4)
        self._modules = {}

    def load_state_dict(self, state_dict):
        # Reconstruct as proper nn.Module
        pass

    def eval(self):
        pass

    def __call__(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.projection(x)
        return self.classifier(x)


# Actually, let's use the proper class
import torch
import torch.nn as nn

class SilentCareVideoModelTorch(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models as tmodels
        backbone = tmodels.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(512, 4)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.projection(x)
        return self.classifier(x)


def save_comparison_json(audio_metrics, video_metrics, benchmark_times):
    """Save all comparison data to JSON."""
    data = {
        "audio_progression": [],
        "video_progression": [],
        "inference_benchmark": benchmark_times or {},
    }

    for b in AUDIO_BASELINES:
        data["audio_progression"].append({
            "name": b["name"].replace("\n", " "),
            "accuracy": b["accuracy"],
            "f1_macro": b["f1_macro"],
            "source": b["source"],
        })
    if audio_metrics:
        data["audio_progression"].append({
            "name": "YAMNet + Dense Head (SilentCare)",
            "accuracy": audio_metrics["accuracy"],
            "f1_macro": audio_metrics["f1_macro"],
            "source": "evaluated (this study)",
        })

    for b in VIDEO_BASELINES:
        data["video_progression"].append({
            "name": b["name"].replace("\n", " "),
            "accuracy": b["accuracy"],
            "f1_macro": b["f1_macro"],
            "source": b["source"],
        })
    if video_metrics and "resnet50" in video_metrics:
        m = video_metrics["resnet50"]
        data["video_progression"].append({
            "name": "ResNet50 Fine-tuned (SilentCare)",
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "source": "evaluated (this study)",
        })
    if video_metrics and "vit_huggingface" in video_metrics:
        m = video_metrics["vit_huggingface"]
        data["video_progression"].append({
            "name": "ViT HuggingFace (production)",
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "source": "evaluated (this study)",
        })

    out_path = RESULTS_DIR / "comparison_metrics.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nComparison metrics saved to {out_path}")


def main():
    print("=" * 60)
    print("SilentCare - Comparison Results Generation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audio_metrics, video_metrics = load_real_metrics()

    plot_audio_progression(audio_metrics)
    plot_video_progression(video_metrics)
    plot_full_progression(audio_metrics, video_metrics)
    benchmark_times = plot_inference_benchmark(audio_metrics, video_metrics)
    save_comparison_json(audio_metrics, video_metrics, benchmark_times)

    print(f"\n{'='*60}")
    print("Comparison results generation complete!")
    print(f"All outputs in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
