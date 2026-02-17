# src/run_single_stroke.py
from src.utils import load_rgba, save_video
from src.diffusion_engine import InkDiffusionSimulator

def main():
    sim = InkDiffusionSimulator(paper_texture_path="data/textures/xuan_paper.jpg")
    stroke = load_rgba("data/strokes/stroke_01.png")
    frames = sim.simulate_single_stroke(stroke, total_frames=40)
    save_video(frames, "output/single_stroke_preview.mp4", fps=12)
    print("[OK] output/single_stroke_preview.mp4")

if __name__ == "__main__":
    main()