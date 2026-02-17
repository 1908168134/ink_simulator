# src/api.py
from flask import Flask, request, jsonify, send_file
from src.diffusion_engine import InkDiffusionSimulator
import os

app = Flask(__name__)
sim = InkDiffusionSimulator(paper_texture_path="data/textures/xuan_paper.jpg")

@app.route("/render", methods=["GET"])
def render():
    """
    参数:
        strokes_dir: 笔画文件夹（相对路径或绝对路径）
        fps: 视频帧率（默认 10）
    返回:
        JSON { video_path, metadata_path }
    """
    strokes_dir = request.args.get("strokes_dir", "data/strokes")
    fps = int(request.args.get("fps", 10))

    video_path, n_strokes, duration = sim.generate_full_process(
        stroke_folder=strokes_dir,
        output_video="output/api_result.mp4",
        fps=fps
    )

    # 生成元数据
    meta = {
        "video_path": os.path.abspath(video_path),
        "total_strokes": n_strokes,
        "duration_sec": round(duration, 2)
    }
    meta_path = "output/api_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return jsonify({
        "video_path": meta["video_path"],
        "metadata_path": os.path.abspath(meta_path)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)