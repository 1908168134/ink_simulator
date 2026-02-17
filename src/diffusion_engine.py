# src/diffusion_engine.py (W4 update)
import cv2
import numpy as np
from .utils import load_texture, alpha_to_mask, add_text

class InkDiffusionSimulator:
    def __init__(self, paper_texture_path: str = None):
        self.paper_path = paper_texture_path
        self.paper_cache = {}

    def _get_paper(self, hw):
        if hw not in self.paper_cache:
            self.paper_cache[hw] = load_texture(self.paper_path, hw)
        return self.paper_cache[hw]

    def simulate_single_stroke(self, stroke_rgba, total_frames=40,
                               stroke_index=1, add_tip=True):
        h, w = stroke_rgba.shape[:2]

        paper_u8 = self._get_paper((h, w))
        paper = paper_u8.astype(np.float32)

        # 纸纹归一化因子：让墨色保留“纸的颗粒”
        paper_gray = cv2.cvtColor(paper_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        paper_norm = 0.88 + 0.20 * paper_gray  # 大约 [0.88, 1.08]
        paper_norm3 = np.repeat(paper_norm[..., None], 3, axis=2)

        alpha = alpha_to_mask(stroke_rgba[..., 3])  # 0~1

        frames = []
        for t in range(total_frames):
            # 1) 扩散核：sigma递增
            sigma = 0.8 + 0.15 * t

            # 2) 膨胀核：随时间变大（但别太猛）
            k = max(3, int(1 + 0.08 * t))
            k = k if k % 2 == 1 else k + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            a_blur = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
            a_dil = cv2.dilate(a_blur, kernel)
            a_dil = np.clip(a_dil, 0.0, 1.0)

            # 3) 墨分层次：深->浅（焦浓淡清的“感觉”）
            # 你可以理解为：时间越晚，墨越淡（但仍保底）
            intensity = np.clip(0.90 - 0.015 * t, 0.30, 0.90)

            # 墨色：越“浓”越接近黑（值小）
            ink_gray = 10 + (1 - intensity) * 35  # 10~45
            ink = np.full((h, w, 3), ink_gray, dtype=np.float32)

            # 4) 合成（带纸纹）
            a3 = np.repeat(a_dil[..., None], 3, axis=2)

            cur = paper * (1 - a3) + ink * a3
            cur = cur * paper_norm3  # 关键：纸纹“压”在墨色上
            cur = np.clip(cur, 0, 255).astype(np.uint8)

            # 5) 教学文字：中间帧叠加
            if add_tip and t == total_frames // 2:
                cur = add_text(cur, f"第{stroke_index}笔：中锋用笔", pos=(30, 40),
                               font_scale=1.0, color=(230, 230, 230), thickness=2)

            frames.append(cur)

        return frames