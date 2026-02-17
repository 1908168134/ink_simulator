# src/make_compare_single.py
import cv2
import numpy as np
from src.utils import load_rgba, ensure_dir
from src.diffusion_engine import InkDiffusionSimulator

def main():
    ensure_dir("output/compare")
    sim = InkDiffusionSimulator("data/textures/xuan_paper.jpg")
    stroke = load_rgba("data/strokes/stroke_01.png")
    h, w = stroke.shape[:2]

    paper = sim._get_paper((h, w))
    alpha = stroke[..., 3].astype(np.float32) / 255.0
    a3 = np.repeat(alpha[..., None], 3, axis=2)

    # no_diffusion: 直接贴（没有扩散/膨胀/纸纹参与）
    ink = np.full((h, w, 3), 20, dtype=np.float32)
    no_diff = paper.astype(np.float32) * (1 - a3) + ink * a3
    no_diff = np.clip(no_diff, 0, 255).astype(np.uint8)

    frames = sim.simulate_single_stroke(stroke, total_frames=36, stroke_index=1, add_tip=False)
    with_diff = frames[-1]

    cv2.imwrite("output/compare/no_diffusion.png", no_diff)
    cv2.imwrite("output/compare/with_diffusion.png", with_diff)
    print("[OK] output/compare/*.png")

if __name__ == "__main__":
    main()