# src/utils.py
import os
from pathlib import Path
import cv2
import numpy as np

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_rgba(path: str, size=None):
    """读取 PNG，保持 Alpha 通道，若 size=(w,h) 给出则 resize"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # (H,W,4)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Image must be RGBA (4 channels), got {img.shape}")

    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def alpha_to_mask(alpha_u8: np.ndarray):
    """Alpha(0-255) -> mask(0-1) + 简单净化"""
    a = alpha_u8.astype(np.float32) / 255.0
    # 去掉很浅的边缘灰（视你的素材情况调整阈值）
    a[a < 0.03] = 0.0
    a[a > 1.0] = 1.0
    return a

def load_texture(path: str, target_hw):
    """加载宣纸纹理或生成噪声纹理，返回 BGR uint8"""
    h, w = target_hw
    if path and Path(path).exists():
        tex = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_AREA)
    else:
        # 低幅度噪声
        tex = np.random.randint(235, 256, (h, w), dtype=np.uint8)
        tex = cv2.GaussianBlur(tex, (0, 0), 1.2)

    return cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)

def add_text(frame, text, pos=(30, 40), font_scale=1.0,
             color=(220, 220, 220), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return frame

def save_video(frames, out_path, fps=10):
    ensure_dir(os.path.dirname(out_path) or ".")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()