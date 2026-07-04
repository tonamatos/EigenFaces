"""
generate_video.py

Animate my_face.png walking along the φ₂ eigenface direction.

Outputs (written to app/output/):
  eigenface_walk.mp4   — face-only animation, ±2.5σ along φ₂, 4 loops × 2 s
  eigenface_split.mp4  — split screen: face (left) + fancy R³ vector (right)

Requirements:
  pip install imageio imageio-ffmpeg pillow numpy matplotlib scikit-learn
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from PIL import Image
import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_faces, compute_mean_face, center_data, compute_pca_via_svd

# ─── config ────────────────────────────────────────────────────────────────────
FPS        = 30
LOOP_SEC   = 2.0   # seconds per back-and-forth sweep
N_LOOPS    = 4     # complete cycles in the video
ALPHA_MAX  = 2.5   # ± standard deviations
EF_IDX     = 1     # φ₂ (0-indexed)
PANEL_PX   = 320   # width = height of each panel in pixels

# Colour palette from app.py
BG     = "#0d1117"
CYAN   = "#00e5ff"
YELLOW = "#ffd93d"
WHITE  = "#e6edf3"
DIM    = "#30363d"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)


# ─── helpers ───────────────────────────────────────────────────────────────────

def load_my_face() -> np.ndarray:
    path = os.path.join(HERE, "my_face.png")
    img  = Image.open(path).convert("L").resize((64, 64), Image.LANCZOS)
    return np.array(img, dtype=np.float64).flatten() / 255.0


def build_alphas() -> np.ndarray:
    """Cosine sweep: −ALPHA_MAX → +ALPHA_MAX → −ALPHA_MAX, tiled N_LOOPS times."""
    n   = int(FPS * LOOP_SEC)
    t   = np.linspace(0, 2 * np.pi, n, endpoint=False)
    one = -ALPHA_MAX * np.cos(t)
    return np.tile(one, N_LOOPS)


def fig_to_frame(fig: plt.Figure) -> np.ndarray:
    """Render figure → PANEL_PX × PANEL_PX uint8 RGB array."""
    fig.canvas.draw()
    buf   = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    frame = np.array(Image.fromarray(buf).resize((PANEL_PX, PANEL_PX), Image.LANCZOS))
    return frame


def write_mp4(frames: list, path: str) -> None:
    writer = imageio.get_writer(
        path, fps=FPS, codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-crf", "20"],
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  saved -> {path}")


# ─── face panel ────────────────────────────────────────────────────────────────

def make_face_frames(my_face, eigenfaces, eigenvalues, alphas) -> list:
    phi = eigenfaces[EF_IDX]
    std = float(np.sqrt(eigenvalues[EF_IDX]))

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100, facecolor=BG)
    frames  = []

    for alpha in alphas:
        modified = np.clip(my_face + alpha * std * phi, 0, 1)
        ax.clear()
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.imshow(modified.reshape(64, 64), cmap="gray",
                  vmin=0, vmax=1, interpolation="bilinear")

        sign = "+" if alpha >= 0 else ""
        ax.set_title(
            f"f̂  =  f  {sign}  {sign}{alpha:.2f}·√λ₂·φ₂",
            color=CYAN, fontfamily="monospace", fontsize=9, pad=6,
        )
        fig.tight_layout(pad=0.5)
        frames.append(fig_to_frame(fig))

        if (len(frames) % 30) == 0:
            print(f"    face frame {len(frames)}/{len(alphas)}")

    plt.close(fig)
    return frames


# ─── vector panel ──────────────────────────────────────────────────────────────

def make_vector_frames(alphas) -> list:
    """
    Fancy R³ illustration: a vector with fixed φ₁, φ₃ components
    and oscillating φ₂ component, shown in a rotating eigenspace.
    (Illustrative only — not real eigenface coordinates.)
    """
    fig = plt.figure(figsize=(3.5, 3.5), dpi=100, facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(BG)

    n_frames  = len(alphas)
    trail_len = 20
    trail: list = []
    frames = []

    for fi, alpha in enumerate(alphas):
        t  = alpha / ALPHA_MAX    # ∈ [−1, 1]

        # Illustrative vector components in eigenface space
        vx = 0.82                 # fixed φ₁ projection
        vy = alpha * 0.40         # φ₂ oscillates with alpha
        vz = 0.30                 # fixed φ₃ projection

        trail.append((vx, vy, vz))
        if len(trail) > trail_len:
            trail.pop(0)

        ax.clear()
        ax.set_facecolor(BG)
        fig.patch.set_facecolor(BG)

        # Pane styling
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor(DIM)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        for ax_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
            ax_obj.line.set_color(DIM)

        # Axis arrows and labels
        AL = 1.2
        for (dx, dy, dz), lbl, col in [
            ((AL, 0, 0), "φ₁", WHITE),
            ((0, AL, 0), "φ₂", CYAN),
            ((0, 0, AL), "φ₃", YELLOW),
        ]:
            ax.quiver(0, 0, 0, dx, dy, dz, color=col, lw=0.8,
                      arrow_length_ratio=0.12, alpha=0.65)
            ax.text(dx * 1.18, dy * 1.18, dz * 1.18, lbl,
                    color=col, fontsize=9, fontfamily="monospace",
                    ha="center", va="center")

        # Origin dot
        ax.scatter([0], [0], [0], color=WHITE, s=10, zorder=5, alpha=0.6)

        # Fading trail of past vector tips
        if len(trail) >= 2:
            th = np.array(trail)
            nT = len(th)
            for j in range(nT - 1):
                frac = (j + 1) / nT
                ax.plot(
                    th[j:j+2, 0], th[j:j+2, 1], th[j:j+2, 2],
                    color=(0.0, frac * 0.85, 1.0 - frac * 0.25),
                    alpha=frac * 0.60, linewidth=1.0,
                )

        # Projection dashed lines (show vector components)
        ax.plot([0, vx], [0, 0],  [0, 0],   color=WHITE,  lw=0.5, ls="--", alpha=0.30)
        ax.plot([0, 0],  [0, vy], [0, 0],   color=CYAN,   lw=0.6, ls="--", alpha=0.55)
        ax.plot([0, 0],  [0, 0],  [0, vz],  color=YELLOW, lw=0.5, ls="--", alpha=0.30)
        # "stalk" connecting foot of φ₂ projection to vector tip
        ax.plot([vx, vx], [0, vy],   [0, 0],   color=CYAN,   lw=0.5, ls=":", alpha=0.40)
        ax.plot([vx, vx], [vy, vy],  [0, vz],  color=YELLOW, lw=0.5, ls=":", alpha=0.30)

        # Main vector — colour shifts cool↔warm with alpha
        vcol = plt.cm.cool(0.5 + 0.5 * t)
        ax.quiver(0, 0, 0, vx, vy, vz,
                  color=vcol, lw=2.2, arrow_length_ratio=0.14)
        ax.scatter([vx], [vy], [vz], color=vcol, s=24, zorder=10)

        # Annotations
        sign = "+" if alpha >= 0 else ""
        ax.text2D(0.04, 0.93,
                  f"v = φ₁ + {sign}{alpha:.2f}σ·φ₂ + 0.3φ₃",
                  transform=ax.transAxes,
                  color=CYAN, fontsize=7.5, fontfamily="monospace")
        ax.text2D(0.04, 0.05,
                  f"‖v‖ = {np.sqrt(vx**2 + vy**2 + vz**2):.3f}",
                  transform=ax.transAxes,
                  color=DIM, fontsize=7.0, fontfamily="monospace")

        ax.set_xlim(-0.15, 1.55)
        ax.set_ylim(-1.30, 1.30)
        ax.set_zlim(-0.15, 1.55)

        # Slow azimuth rotation for depth
        azim = 32 + 14 * np.sin(2 * np.pi * fi / n_frames)
        ax.view_init(elev=20, azim=azim)
        fig.tight_layout(pad=0.2)
        frames.append(fig_to_frame(fig))

        if (len(frames) % 30) == 0:
            print(f"    vector frame {len(frames)}/{n_frames}")

    plt.close(fig)
    return frames


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading Olivetti faces & computing PCA …")
    X, _, _          = load_faces()
    mean_face        = compute_mean_face(X)
    X_centered       = center_data(X, mean_face)
    evals, efaces, _ = compute_pca_via_svd(X_centered, n_components=10)

    print("Loading my_face.png …")
    my_face = load_my_face()

    alphas = build_alphas()
    total  = len(alphas)
    secs   = total / FPS
    print(f"Sequence: {total} frames  ({secs:.1f}s at {FPS}fps, {N_LOOPS} × {LOOP_SEC}s loops)")

    # ── Video 1: face only ────────────────────────────────────────────────────
    print("\nRendering face frames …")
    face_frames = make_face_frames(my_face, efaces, evals, alphas)
    print("Writing eigenface_walk.mp4 …")
    write_mp4(face_frames, os.path.join(OUT, "eigenface_walk.mp4"))

    # ── Video 2: split screen ─────────────────────────────────────────────────
    print("\nRendering vector frames …")
    vec_frames = make_vector_frames(alphas)

    print("Writing eigenface_split.mp4 …")
    # 320 + 320 = 640px wide — clean multiple of 16 for H.264
    split_frames = [
        np.concatenate([l, r], axis=1)
        for l, r in zip(face_frames, vec_frames)
    ]
    write_mp4(split_frames, os.path.join(OUT, "eigenface_split.mp4"))

    print("\nDone.")


if __name__ == "__main__":
    main()
