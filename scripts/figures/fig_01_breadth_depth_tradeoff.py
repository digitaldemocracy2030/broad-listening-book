"""第1章 「深さと規模のトレードオフ」概念図(模式図)を生成する。

両軸ともにリニアスケールで、反比例曲線(深さ × 規模 = 一定)を描き、
2つの代表的な手法(深いがインタビュー / 広いがアンケート)を曲線上の対極の点で示し、
曲線の右上の領域を「ブロードリスニングが目指す領域」としてハイライトする。

具体的な数値は描かない模式図とし、軸目盛は出さず方向のみを示す。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

matplotlib.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "Noto Sans JP",
    "MS Gothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)

# 反比例曲線 y = C / x (linear-linear axes)
C = 1.0
x = np.linspace(0.16, 10, 400)
y = C / x

# 曲線
ax.plot(x, y, color="#444", lw=1.8, zorder=2)

# ブロードリスニング目指す領域を楕円(丸)で囲む
target_ellipse = Ellipse(
    xy=(4.5, 4.6),
    width=4.0,
    height=2.6,
    fill=True,
    facecolor="#ffd84d",
    alpha=0.22,
    edgecolor="#a06000",
    linewidth=1.8,
    linestyle="--",
    zorder=3,
)
ax.add_patch(target_ellipse)

# 曲線上の2つの代表点
pt_a = (0.28, 3.57)  # 深いが人数少ない (インタビュー型) - y = 1/0.28
pt_b = (5.0, 0.20)   # 広いが浅い (アンケート型) - y = 1/5
for px, py in [pt_a, pt_b]:
    ax.scatter(
        [px],
        [py],
        s=70,
        color="#d2322f",
        zorder=5,
        edgecolor="white",
        linewidth=1.2,
    )

# 点A: 深いが人数少ない - 右に注釈
ax.annotate(
    "少人数に\n深くじっくり聴く\n(インタビューなど)",
    xy=pt_a,
    xytext=(1.7, 2.6),
    fontsize=8.5,
    ha="center",
    va="center",
    arrowprops=dict(arrowstyle="-", color="#888", lw=0.7),
)

# 点B: 広いが浅い - 右に注釈
ax.annotate(
    "大人数に\n短く聴く\n(アンケートなど)",
    xy=pt_b,
    xytext=(6.5, 1.3),
    fontsize=8.5,
    ha="center",
    va="center",
    arrowprops=dict(arrowstyle="-", color="#888", lw=0.7),
)

# 曲線のラベル(インライン、曲線の上側に配置)
ax.text(
    2.7,
    1.2,
    "従来手法のトレードオフ曲線\n(深さ × 規模 = 一定)",
    fontsize=8.5,
    color="#333",
    ha="center",
    va="center",
)

# 目指す領域ラベル(楕円の中央に配置)
ax.text(
    4.5,
    4.6,
    "ブロードリスニングが\n目指す領域\n(広く × 深く)",
    fontsize=10.5,
    ha="center",
    va="center",
    color="#a06000",
    weight="bold",
    zorder=4,
)

# 軸: リニア、目盛なし、矢印的なラベル
ax.set_xlim(0, 7.6)
ax.set_ylim(0, 6.4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("規模 (聴く人数) →", fontsize=9.5, labelpad=4)
ax.set_ylabel("深さ (一人あたり対話時間) →", fontsize=9.5, labelpad=4)
ax.set_title("深さと規模のトレードオフ", fontsize=11, pad=8)

# 上 と 右の枠線を消して開放的に
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

plt.tight_layout()
out = Path("images/01_breadth_depth_tradeoff.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"saved {out}")
