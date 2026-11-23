"""Draw the Serial Pyramid DINOv3 Adapter architecture as a vector PDF."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name(
    "serial_pyramid_dinov3_adapter_architecture.pdf")
PREVIEW = Path("/tmp/serial_pyramid_dinov3_adapter_architecture.png")


COLORS = {
    "swin": "#DCEBFA",
    "dino": "#FDE7D9",
    "adapter": "#E6DDF4",
    "scale": "#FFF1BE",
    "fusion": "#DDF2E3",
    "output": "#D8EEF0",
    "edge": "#263746",
    "muted": "#647482",
}


def box(ax, x, y, w, h, text, color, *, fontsize=9, lw=1.25,
        radius=0.025, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=color,
        edgecolor=COLORS["edge"],
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color="#17232D", weight=weight, linespacing=1.25, zorder=4,
    )
    return patch


def arrow(ax, start, end, *, color=None, lw=1.35, style="-|>",
          connectionstyle="arc3", zorder=2):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        color=color or COLORS["edge"],
        connectionstyle=connectionstyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        8, 8.68, "Serial Pyramid DINOv3 Adapter",
        ha="center", va="center", fontsize=20, weight="bold",
        color="#17232D",
    )
    ax.text(
        8, 8.34,
        "Swin spatial pyramid guides frozen DINOv3 features through four serial deformable cross-attention stages",
        ha="center", va="center", fontsize=10.5, color=COLORS["muted"],
    )

    # Input and two encoders.
    box(ax, 0.35, 3.75, 1.05, 1.05, "Input image\n1536 × 1536", "#EFF2F4",
        fontsize=9.5, weight="bold")
    arrow(ax, (1.40, 4.27), (1.82, 6.23))
    arrow(ax, (1.40, 4.27), (1.82, 2.08))

    box(ax, 1.85, 5.82, 1.55, 0.82, "Swin-Tiny\ntrainable", COLORS["swin"],
        fontsize=10, weight="bold")
    box(ax, 1.85, 1.67, 1.55, 0.82, "DINOv3-S/16\nfrozen", COLORS["dino"],
        fontsize=10, weight="bold")

    # Swin and DINO stage features.
    xs = [4.05, 7.05, 10.05, 13.05]
    swin_text = [
        "S₁\n1/4 · 96 ch", "S₂\n1/8 · 192 ch",
        "S₃\n1/16 · 384 ch", "S₄\n1/32 · 768 ch",
    ]
    dino_text = [
        "D₁ · block 3\n1/16 · 384 ch", "D₂ · block 6\n1/16 · 384 ch",
        "D₃ · block 9\n1/16 · 384 ch", "D₄ · block 12\n1/16 · 384 ch",
    ]
    for i, x in enumerate(xs):
        box(ax, x, 6.75, 1.72, 0.68, swin_text[i], COLORS["swin"], fontsize=9)
        box(ax, x, 0.72, 1.72, 0.68, dino_text[i], COLORS["dino"], fontsize=9)
        if i == 0:
            arrow(ax, (3.40, 6.23), (x, 7.09))
            arrow(ax, (3.40, 2.08), (x, 1.06))
        else:
            arrow(ax, (xs[i - 1] + 1.72, 7.09), (x, 7.09),
                  color=COLORS["muted"], lw=1.0)
            arrow(ax, (xs[i - 1] + 1.72, 1.06), (x, 1.06),
                  color=COLORS["muted"], lw=1.0)

    # Stage 1 scale preparation.
    box(ax, 4.25, 5.53, 1.32, 0.55, "1×1 Conv\n→ 384 ch", COLORS["scale"], fontsize=8.5)
    box(ax, 4.25, 4.70, 1.32, 0.55, "Patch embed\n2×2, stride 2", COLORS["scale"], fontsize=8.2)
    arrow(ax, (4.91, 6.75), (4.91, 6.08))
    arrow(ax, (4.91, 5.53), (4.91, 5.25))

    # Attention blocks and serial state stream.
    ca_y, ca_w, ca_h = 3.42, 2.05, 0.84
    ca_labels = [
        "CA₁ + CFFN\nquery: 1/8",
        "CA₂ + CFFN\nquery: 1/8",
        "CA₃ + CFFN\nquery: 1/16",
        "CA₄ + CFFN\nquery: 1/32",
    ]
    for i, x in enumerate(xs):
        ca_x = x - 0.16
        box(ax, ca_x, ca_y, ca_w, ca_h, ca_labels[i], COLORS["adapter"],
            fontsize=9.2, weight="bold")
        # DINO tokens are values in deformable cross attention.
        arrow(ax, (x + 0.86, 1.40), (x + 0.86, ca_y), color="#C16F3D")
        ax.text(x + 0.98, 2.25, "value", fontsize=7.7, color="#9A572F",
                rotation=90, ha="left", va="center")

    arrow(ax, (4.91, 4.70), (4.91, ca_y + ca_h))

    # Swin residual injection for stages 2-4.
    for i in range(1, 4):
        x = xs[i]
        box(ax, x + 0.18, 5.38, 1.36, 0.55, "1×1 Conv\n→ 384 ch",
            COLORS["scale"], fontsize=8.5)
        arrow(ax, (x + 0.86, 6.75), (x + 0.86, 5.93))
        circle_x, circle_y = x + 0.86, 4.88
        ax.scatter([circle_x], [circle_y], s=340, facecolors=COLORS["fusion"],
                   edgecolors=COLORS["edge"], linewidths=1.2, zorder=4)
        ax.text(circle_x, circle_y, "+", ha="center", va="center",
                fontsize=15, weight="bold", color=COLORS["edge"], zorder=5)
        arrow(ax, (circle_x, 5.38), (circle_x, 5.10))
        arrow(ax, (circle_x, 4.66), (circle_x, ca_y + ca_h))

    # Serial transitions CA1→CA2, CA2→CA3, CA3→CA4.
    # CA1 and CA2 have equal resolution; later transitions downsample.
    transition_labels = ["identity", "2×2 / s2", "2×2 / s2"]
    for i in range(3):
        x0 = xs[i] - 0.16 + ca_w
        x1 = xs[i + 1] - 0.16
        mid = (x0 + x1) / 2
        # Route the state upward into the residual add node.
        target = (xs[i + 1] + 0.86 - 0.20, 4.88)
        arrow(ax, (x0, ca_y + ca_h / 2), target,
              connectionstyle="arc3,rad=-0.18")
        ax.text(mid, 4.38, transition_labels[i], ha="center", va="center",
                fontsize=7.7, color=COLORS["muted"])

    # Output extraction before the next Swin residual and raw ViT residual.
    out_y = 2.02
    output_text = [
        "F₁\n1/4 · 384 ch", "F₂\n1/8 · 384 ch",
        "F₃\n1/16 · 384 ch", "F₄\n1/32 · 384 ch",
    ]
    for i, x in enumerate(xs):
        out_x = x + 0.06
        if i == 0:
            scale_label = "Deconv ×2"
        else:
            scale_label = ""
        # Vertical branch from CA output to feature output.
        arrow(ax, (x + 0.86, ca_y), (x + 0.86, out_y + 0.63),
              color="#4C667A")
        if scale_label:
            ax.text(x + 0.98, 2.86, scale_label, fontsize=7.5,
                    color=COLORS["muted"], rotation=90, va="center")
        # DINO residual is already represented by the upward value arrow;
        # show a separate dashed residual path into the output fusion.
        arrow(ax, (x + 0.46, 1.40), (x + 0.46, out_y),
              color="#C16F3D", lw=1.0, style="-|>",
              connectionstyle="arc3,rad=0.25", zorder=1)
        ax.text(x + 0.08, 1.66, "resize + add", fontsize=7.2,
                color="#9A572F", ha="left")
        box(ax, out_x, out_y, 1.60, 0.63, output_text[i], COLORS["output"],
            fontsize=8.8, weight="bold")

    # Decoder summary.
    decoder_x, decoder_y = 6.15, 7.78
    box(ax, decoder_x, decoder_y, 3.70, 0.44,
        "Four-level pyramid {F₁, F₂, F₃, F₄} → MultiHeads decoder",
        "#E7F3EA", fontsize=9.2, weight="bold")

    # Legend and implementation notes.
    ax.text(0.45, 0.35,
            "CA: multi-scale deformable cross-attention (6 heads, 4 sampling points)   ·   "
            "CFFN ratio: 0.25   ·   deform ratio: 0.5   ·   add_vit_feature: True",
            fontsize=8.2, color=COLORS["muted"], ha="left", va="center")

    fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(PREVIEW, format="png", dpi=140, bbox_inches="tight",
                pad_inches=0.12)
    print(OUT)


if __name__ == "__main__":
    main()
