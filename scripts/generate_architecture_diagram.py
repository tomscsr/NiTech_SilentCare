"""
SilentCare - Generate Architecture Diagram
===========================================
Produces a publication-ready architecture diagram showing the full
SilentCare multimodal pipeline.

Usage:
    python scripts/generate_architecture_diagram.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


def draw_box(ax, x, y, w, h, text, color, fontsize=9, fontweight="normal", alpha=0.9, text_color="white"):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", style="->", lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=1,
    )


def draw_label(ax, x, y, text, fontsize=7, color="#666666"):
    """Draw a small label."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic", zorder=3)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Colors
    C_INPUT = "#34495e"
    C_AUDIO = "#3498db"
    C_VIDEO = "#e74c3c"
    C_FUSION = "#2ecc71"
    C_OUTPUT = "#9b59b6"
    C_DB = "#f39c12"
    C_FEEDBACK = "#1abc9c"

    # Title
    ax.text(8, 9.6, "SilentCare - Multimodal Emotion Monitoring Architecture",
            ha="center", fontsize=16, fontweight="bold", color="#2c3e50")

    # === INPUT LAYER ===
    draw_box(ax, 2.5, 8.3, 2.2, 0.7, "Microphone\n(22050 Hz mono)", C_INPUT, fontsize=9, fontweight="bold")
    draw_box(ax, 5.5, 8.3, 2.2, 0.7, "Camera / Video\n(RGB frames)", C_INPUT, fontsize=9, fontweight="bold")

    # === AUDIO PIPELINE (left) ===
    draw_box(ax, 2.5, 7.0, 2.5, 0.6, "YAMNet (TF Hub, frozen)\n1024-dim embeddings/frame", C_AUDIO, fontsize=8)
    draw_arrow(ax, 2.5, 7.95, 2.5, 7.35, C_AUDIO)
    draw_label(ax, 1.5, 7.5, "16kHz\nresample")

    draw_box(ax, 2.5, 6.0, 2.5, 0.6, "Temporal Aggregation\nmean + max + std -> 3072-dim", C_AUDIO, fontsize=8)
    draw_arrow(ax, 2.5, 6.65, 2.5, 6.35, C_AUDIO)

    draw_box(ax, 2.5, 5.0, 2.5, 0.8, "Classification Head\nDense(256) -> BN -> Drop(0.4)\nDense(128) -> Drop(0.3)\nDense(4, softmax)", C_AUDIO, fontsize=7)
    draw_arrow(ax, 2.5, 5.65, 2.5, 5.45, C_AUDIO)

    draw_box(ax, 2.5, 3.8, 2.0, 0.5, "Audio Probabilities\n[P_d, P_a, P_al, P_c]", C_AUDIO, fontsize=8, alpha=0.7)
    draw_arrow(ax, 2.5, 4.55, 2.5, 4.1, C_AUDIO)

    # === VIDEO PIPELINE (right) ===
    draw_box(ax, 5.5, 7.0, 2.5, 0.6, "Face Detection\n(Haar + CLAHE)", C_VIDEO, fontsize=8)
    draw_arrow(ax, 5.5, 7.95, 5.5, 7.35, C_VIDEO)

    draw_box(ax, 5.5, 6.0, 2.5, 0.6, "ViT (HuggingFace)\ntrpakov/vit-face-expression", C_VIDEO, fontsize=8)
    draw_arrow(ax, 5.5, 6.65, 5.5, 6.35, C_VIDEO)

    draw_box(ax, 5.5, 5.0, 2.5, 0.6, "7-class FER -> 4-class\nSilentCare Mapping", C_VIDEO, fontsize=8)
    draw_arrow(ax, 5.5, 5.65, 5.5, 5.35, C_VIDEO)

    draw_box(ax, 5.5, 3.8, 2.0, 0.5, "Video Probabilities\n[P_d, P_a, P_al, P_c]", C_VIDEO, fontsize=8, alpha=0.7)
    draw_arrow(ax, 5.5, 4.35, 5.5, 4.1, C_VIDEO)

    # === FUSION ===
    draw_box(ax, 4.0, 2.7, 3.8, 0.8,
             "Adaptive Weighted Fusion\n0.30 x Audio + 0.70 x Video\n+ Agreement Boost (x1.3) + Uncertainty Gate",
             C_FUSION, fontsize=8, fontweight="bold")
    draw_arrow(ax, 2.5, 3.5, 3.0, 3.15, C_AUDIO, lw=2)
    draw_arrow(ax, 5.5, 3.5, 5.0, 3.15, C_VIDEO, lw=2)
    draw_label(ax, 2.2, 3.3, "w=0.30")
    draw_label(ax, 5.8, 3.3, "w=0.70")

    # === OUTPUT ===
    draw_box(ax, 4.0, 1.5, 3.0, 0.6,
             "Prediction: DISTRESS | ANGRY | ALERT | CALM\n+ Confidence Score",
             C_OUTPUT, fontsize=9, fontweight="bold")
    draw_arrow(ax, 4.0, 2.25, 4.0, 1.85, C_FUSION, lw=2)

    # === ALERT SYSTEM (right side) ===
    draw_box(ax, 9.5, 2.7, 2.5, 0.6, "Alert Manager\nThreshold + Cooldown", C_OUTPUT, fontsize=8)
    draw_arrow(ax, 6.0, 2.7, 8.2, 2.7, C_OUTPUT)

    draw_box(ax, 9.5, 1.5, 2.5, 0.6, "Notification\n(Dashboard SSE)", C_OUTPUT, fontsize=8, alpha=0.7)
    draw_arrow(ax, 9.5, 2.35, 9.5, 1.85, C_OUTPUT)

    # === DATABASE (bottom right) ===
    draw_box(ax, 13.0, 2.7, 2.2, 0.6, "SQLite Database\nSessions + Events", C_DB, fontsize=8, fontweight="bold")
    draw_arrow(ax, 10.8, 2.7, 11.85, 2.7, C_DB)

    # === FEEDBACK LOOP ===
    draw_box(ax, 13.0, 1.5, 2.2, 0.6, "Human-in-the-Loop\nFeedback Service", C_FEEDBACK, fontsize=8)
    draw_arrow(ax, 13.0, 2.35, 13.0, 1.85, C_FEEDBACK)

    draw_box(ax, 13.0, 0.5, 2.2, 0.5, "Fine-tuning Pipeline\n(Incremental)", C_FEEDBACK, fontsize=7, alpha=0.7)
    draw_arrow(ax, 13.0, 1.15, 13.0, 0.8, C_FEEDBACK)

    # Feedback arrow back to models
    ax.annotate(
        "", xy=(2.5, 4.55), xytext=(11.85, 0.5),
        arrowprops=dict(arrowstyle="->", color=C_FEEDBACK, lw=1.2,
                        connectionstyle="arc3,rad=-0.3", linestyle="dashed"),
        zorder=1,
    )
    draw_label(ax, 7.5, 0.3, "model update (retrain)")

    # === DASHBOARD ===
    draw_box(ax, 9.5, 8.3, 3.0, 0.7, "Flask Dashboard\n(Real-time + Offline)", C_OUTPUT, fontsize=9, fontweight="bold")

    draw_box(ax, 9.5, 7.0, 2.5, 0.6, "SSE Stream\n/api/events", C_OUTPUT, fontsize=8, alpha=0.7)
    draw_arrow(ax, 9.5, 7.95, 9.5, 7.35, C_OUTPUT)

    draw_box(ax, 12.5, 7.0, 2.5, 0.6, "Offline Mode\nMP4 Upload + FFmpeg", C_OUTPUT, fontsize=8, alpha=0.7)
    draw_arrow(ax, 10.5, 7.95, 12.0, 7.35, C_OUTPUT)

    # === TRAINING PATH (top right) ===
    draw_box(ax, 13.5, 8.3, 2.2, 0.7, "Training\n(RAF-DB + Audio DS)", C_DB, fontsize=8)

    # === LEGEND ===
    legend_items = [
        mpatches.Patch(facecolor=C_INPUT, label="Input Capture"),
        mpatches.Patch(facecolor=C_AUDIO, label="Audio Pipeline"),
        mpatches.Patch(facecolor=C_VIDEO, label="Video Pipeline"),
        mpatches.Patch(facecolor=C_FUSION, label="Fusion"),
        mpatches.Patch(facecolor=C_OUTPUT, label="Output / Dashboard"),
        mpatches.Patch(facecolor=C_DB, label="Storage / Training"),
        mpatches.Patch(facecolor=C_FEEDBACK, label="Feedback Loop"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              ncol=4, framealpha=0.8, edgecolor="lightgray")

    # 4 class labels at bottom
    class_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    class_names = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        ax.add_patch(FancyBboxPatch(
            (1.5 + i * 1.5, 0.15), 1.2, 0.35,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=1, alpha=0.8,
        ))
        ax.text(2.1 + i * 1.5, 0.32, name, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    plt.tight_layout()
    out_path = RESULTS_DIR / "architecture.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Architecture diagram saved to {out_path}")


if __name__ == "__main__":
    main()
