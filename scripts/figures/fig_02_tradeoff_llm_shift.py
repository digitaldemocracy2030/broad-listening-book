"""第2章 「LLM登場による達成可能領域の拡大」概念図を生成する。

第1章の反比例曲線を2本重ねて、LLM登場前後の差を示す:
- 内側の曲線: LLM前のトレードオフ (低予算で実現可能な領域の境界)
- 外側の曲線: LLM後のトレードオフ (解析コストが下がり境界が外側へ移動)
- 2本の曲線の間: 「LLMが新たに到達可能にした領域」
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "Noto Sans JP",
    "MS Gothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=200)

C_before = 1.0
C_after = 9.0  # 大きく外側に押し上げる
x = np.linspace(0.18, 10, 600)
y_before = C_before / x
y_after = C_after / x

# 2本の曲線の間 = LLMが拡張した領域
y_top = np.minimum(y_after, 6.5)
ax.fill_between(
    x, y_before, y_top,
    where=(y_top > y_before),
    color="#ffd84d", alpha=0.30, zorder=1,
)

# 曲線
ax.plot(x, y_before, color="#888", lw=1.5, zorder=2)
ax.plot(x, y_after, color="#333", lw=1.8, zorder=2)

# 曲線のインラインラベル(右側、各曲線の上に配置)
ax.text(7.0, C_before / 7.0 + 0.25, "LLM登場前",
        fontsize=9.5, color="#666",
        ha="left", va="bottom")
ax.text(7.0, C_after / 7.0 + 0.25, "LLM登場後",
        fontsize=10, color="#222",
        ha="left", va="bottom", weight="bold")

# 矢印: 曲線が外側にシフト(内側曲線上の点 → 外側曲線上の点 を厳密に指す)
arrow_x = 3.0
arrow_y_start = C_before / arrow_x  # = 0.33
arrow_y_end = C_after / arrow_x      # = 3.0
ax.annotate(
    "", xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
    arrowprops=dict(arrowstyle="->", color="#a06000", lw=1.8,
                    shrinkA=2, shrinkB=2),
    zorder=4,
)
ax.text(arrow_x + 0.25, 0.85,
        "コストが下がり\n曲線が外側へシフト",
        fontsize=9, color="#a06000", ha="left", va="center")

# 拡張領域ラベル(中央〜上寄り、曲線と被らない位置)
ax.text(
    5.0, 3.5,
    "LLMが新たに\n到達可能にした領域",
    fontsize=11,
    ha="center", va="center",
    color="#a06000", weight="bold",
)

# 制約条件の注記(図の右上)
ax.text(
    9.7, 5.2,
    "※各曲線はコストが一定のときの上限",
    fontsize=8, color="#555",
    ha="right", va="top", style="italic",
)

# 軸
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("規模 (聴く人数) →", fontsize=9.5, labelpad=4)
ax.set_ylabel("深さ (一人あたり対話時間) →", fontsize=9.5, labelpad=4)
ax.set_title("LLM登場で広がる達成可能領域", fontsize=11, pad=8)

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

plt.tight_layout()
out = Path("images/02_tradeoff_llm_before_after.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"saved {out}")
