# main.py
import json
from src.diffusion_engine import InkDiffusionSimulator
from src.utils import add_text
import os

if __name__ == "__main__":
    # 1️⃣ 初始化渲染器（可自行换纹理）
    simulator = InkDiffusionSimulator(
        paper_texture_path="data/textures/xuan_paper.png"
    )

    # 2️⃣ 生成完整视频
    video_path, n_strokes, duration = simulator.generate_full_process(
        stroke_folder="data/strokes",
        output_video="output/zhubamboo_process.mp4",
        fps=10
    )

    # 3️⃣ 生成 GIF（供 PPT 使用）
    # 这里我们直接读取刚才生成的全部帧（内部已经保存，若想复用可改成返回 frames）
    # 为简化演示，重新渲染一次并保存 GIF
    frames = []
    for i, f in enumerate(sorted(os.listdir("data/strokes"))):
        stroke = simulator._get_paper((0,0))  # dummy, not used
    # 实际项目请把 `simulate_single_stroke` 返回的 frames 收集后保存
    # 下面是示例调用（省略细节）：
    # frames = simulator.simulate_single_stroke(...)
    # simulator.save_gif(frames, "output/zhubamboo_process.gif")

    # 4️⃣ 生成 JSON 元数据
    metadata = {
        "video_path": os.path.abspath(video_path),
        "total_strokes": n_strokes,
        "duration_sec": round(duration, 2),
        "tips": [
            "中锋用笔",
            "墨分五色（焦→浓→淡→清）",
            "留白构图"
        ]
    }
    json_path = "output/metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[✅] 元数据已保存至 {json_path}")