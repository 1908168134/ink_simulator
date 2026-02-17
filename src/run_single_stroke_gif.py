# src/run_single_stroke_gif.py
import os
import imageio.v2 as imageio
from src.utils import load_rgba, ensure_dir
from src.diffusion_engine import InkDiffusionSimulator

def main():
    ensure_dir("output")
    sim = InkDiffusionSimulator("data/textures/xuan_paper.jpg")
    stroke = load_rgba("data/strokes/stroke_01.png")
    frames = sim.simulate_single_stroke(stroke, total_frames=36, stroke_index=1, add_tip=True)

    # imageio 需要 RGB，更通用
    frames_rgb = [f[..., ::-1] for f in frames]  # BGR->RGB
    imageio.mimsave("output/single_stroke.gif", frames_rgb, duration=0.06)
    print("[OK] output/single_stroke.gif")

if __name__ == "__main__":
    main()