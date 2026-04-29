"""第13章 デンドログラム切断による上位クラスタ数の決定図を生成する。

20個の下位クラスタを Ward法で階層的に統合し、デンドログラムをどこで「切る」かで
上位クラスタの数が決まることを可視化。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

matplotlib.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "Noto Sans JP",
    "MS Gothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# Reproducible synthetic data: 20 cluster centers grouped into 5 super-clusters
rng = np.random.default_rng(42)
n_super = 5
n_sub_per_super = 4
super_centers = rng.uniform(-8, 8, size=(n_super, 8))
sub_points = []
for sc in super_centers:
    for _ in range(n_sub_per_super):
        sub_points.append(sc + rng.normal(0, 0.6, size=8))
sub_points = np.array(sub_points)

# Ward linkage
Z = linkage(sub_points, method="ward")

fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=200)

# Color list for 5 super-clusters
super_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

# Draw dendrogram with custom color threshold to get 5 clusters
# Compute cut height that gives exactly 5 clusters
# fcluster equivalent: take heights[-(5-1)] = heights[-4]
heights = sorted(Z[:, 2])
cut_5_height = (heights[-5] + heights[-4]) / 2  # between 5th-from-top and 4th-from-top merges

dendrogram(
    Z,
    ax=ax,
    color_threshold=cut_5_height,
    above_threshold_color="#888888",
    leaf_font_size=11,
    leaf_rotation=0,
    labels=[str(i + 1) for i in range(len(sub_points))],
)

# Draw horizontal cut line for 5 clusters
ax.axhline(y=cut_5_height, color="#dc2626", linestyle="--", lw=2, alpha=0.85)
ax.text(0.5, cut_5_height + 0.2,
        "上位クラスタ数=5 で切る",
        color="#dc2626", fontsize=13, weight="bold", va="bottom")

# Title and labels
ax.set_title("Ward法によるデンドログラム: どこで切るかで上位クラスタ数が決まる",
             fontsize=15, weight="bold", pad=15)
ax.set_xlabel("下位クラスタ番号(K-meansで生成された20個)", fontsize=12)
ax.set_ylabel("クラスタ間の距離(高いほど統合のコストが大きい)", fontsize=12)

# Annotation: arrow + text for "ここで切る → 5つ"
ax.annotate(
    "高いところで切る\n→ 大きな塊だけが残る\n(上位クラスタ少)",
    xy=(2, cut_5_height + 0.3),
    xytext=(2, cut_5_height + 4),
    fontsize=10, color="#374151",
    ha="left",
    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2),
)
ax.annotate(
    "低いところで切る\n→ 細かく分かれる\n(上位クラスタ多)",
    xy=(18, 0.3),
    xytext=(15, 3),
    fontsize=10, color="#374151",
    ha="left",
    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2),
)

# Show the 5 resulting groupings as a band at the bottom
# Find which leaf belongs to which cluster after cut
from scipy.cluster.hierarchy import fcluster
labels = fcluster(Z, t=5, criterion="maxclust")
# dendrogram leaf order
from scipy.cluster.hierarchy import leaves_list
leaf_order = leaves_list(Z)
ordered_labels = labels[leaf_order]

# Draw colored bars below x-axis
ymin = ax.get_ylim()[0]
bar_height = 0.35
for i, lab in enumerate(ordered_labels):
    color = super_colors[(lab - 1) % len(super_colors)]
    ax.add_patch(
        plt.Rectangle(
            (i * 10 + 5 - 5, -bar_height - 0.2),
            10, bar_height,
            color=color, alpha=0.7, clip_on=False,
        )
    )

# Adjust ylim to make room
ax.set_ylim(-1.0, ax.get_ylim()[1])

plt.tight_layout()
out = Path("images/13_dendrogram_cut.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
