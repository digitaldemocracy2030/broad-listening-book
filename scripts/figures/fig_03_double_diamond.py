"""第3章 ダブルダイヤモンドモデル概念図(BL/パブコメ対応版)を生成する。

2つのダイヤモンドを横並びで描き、4フェーズ(Discover/Define/Develop/Deliver)に
それぞれの行動(声を広く集める/論点を絞り込む/解決策を広げる/解決策を絞り込む)を
明記。さらに各「収束」フェーズの直下に、対応するツール
(Define=ブロードリスニング、Deliver=パブリックコメント)を配置する。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

matplotlib.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "Noto Sans JP",
    "MS Gothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=200)
ax.set_xlim(0, 9.2)
ax.set_ylim(0.2, 4.2)
ax.set_aspect("equal")
ax.axis("off")

# Diamond 1: 課題発見・定義 (塗りのみ、エッジは下で矢印として描く)
diamond1 = Polygon(
    [(0.5, 2.0), (2.5, 3.4), (4.5, 2.0), (2.5, 0.6)],
    closed=True, fill=True,
    facecolor="#e3f2fd", ec="none",
)
ax.add_patch(diamond1)

# Diamond 2: 解決策の検討・実施 (塗りのみ、エッジは下で矢印として描く)
diamond2 = Polygon(
    [(4.5, 2.0), (6.5, 3.4), (8.5, 2.0), (6.5, 0.6)],
    closed=True, fill=True,
    facecolor="#fff3cd", ec="none",
)
ax.add_patch(diamond2)


def draw_edge_arrow(x1: float, y1: float, x2: float, y2: float,
                    color: str, lw: float = 2.4) -> None:
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        shrinkA=0, shrinkB=0),
        zorder=3,
    )


def edge_label(x1: float, y1: float, x2: float, y2: float,
               text: str, color: str) -> None:
    """辺の中央にテキストを配置(辺の角度に合わせて回転)"""
    import math
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    ax.text(
        mid_x, mid_y, text,
        rotation=angle, rotation_mode="anchor",
        ha="center", va="center",
        fontsize=10.5, color=color, weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        zorder=4,
    )


# Diamond 1 のエッジを矢印で描き、辺に phase 名を配置
draw_edge_arrow(0.5, 2.0, 2.5, 3.4, "#1a4d7a")  # top-left: 発散↗
draw_edge_arrow(0.5, 2.0, 2.5, 0.6, "#1a4d7a")  # bottom-left: 発散↘
draw_edge_arrow(2.5, 3.4, 4.5, 2.0, "#1a4d7a")  # top-right: 収束↘
draw_edge_arrow(2.5, 0.6, 4.5, 2.0, "#1a4d7a")  # bottom-right: 収束↗

edge_label(0.5, 2.0, 2.5, 3.4, "Discover (発散)", "#1a4d7a")
edge_label(2.5, 3.4, 4.5, 2.0, "Define (収束)", "#1a4d7a")

# Diamond 2 のエッジを矢印で描き、辺に phase 名を配置
draw_edge_arrow(4.5, 2.0, 6.5, 3.4, "#a06000")
draw_edge_arrow(4.5, 2.0, 6.5, 0.6, "#a06000")
draw_edge_arrow(6.5, 3.4, 8.5, 2.0, "#a06000")
draw_edge_arrow(6.5, 0.6, 8.5, 2.0, "#a06000")

edge_label(4.5, 2.0, 6.5, 3.4, "Develop (発散)", "#a06000")
edge_label(6.5, 3.4, 8.5, 2.0, "Deliver (収束)", "#a06000")

# 各フェーズの行動説明をダイヤモンドの上半分中央に配置

diamond_center_y = 2.0

ax.text(1.6, diamond_center_y, "声を広く集める",
        ha="center", va="center", fontsize=9.5, color="#1a4d7a")
ax.text(3.4, diamond_center_y, "論点を絞り込む",
        ha="center", va="center", fontsize=9.5, color="#1a4d7a")
ax.text(5.6, diamond_center_y, "解決策を広げる",
        ha="center", va="center", fontsize=9.5, color="#a06000")
ax.text(7.4, diamond_center_y, "解決策を絞り込む",
        ha="center", va="center", fontsize=9.5, color="#a06000")

# 上部: ダイヤモンドのタイトル
ax.text(2.5, 3.7, "課題発見・課題定義",
        ha="center", va="bottom",
        fontsize=11.5, weight="bold", color="#1a4d7a")
ax.text(6.5, 3.7, "解決策の検討・実施",
        ha="center", va="bottom",
        fontsize=11.5, weight="bold", color="#a06000")

# ダイヤモンド外の課題/解決ラベル
ax.text(0.3, 2.0, "課題",
        ha="right", va="center", fontsize=15, color="#444")
ax.text(8.7, 2.0, "解決",
        ha="left", va="center", fontsize=15, color="#444")

# 収束フェーズに対応するツール (Define = BL, Deliver = パブコメ) を
# 各 収束 半分のダイヤモンド内に埋め込む(矢印より前面に描画)
def add_tool_box_inside(x_center: float, y_center: float,
                         label: str, color: str,
                         box_w: float = 1.6, box_h: float = 0.4) -> None:
    rect = FancyBboxPatch(
        (x_center - box_w / 2, y_center - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.06",
        fc="white", ec=color, lw=1.4,
        zorder=10,
    )
    ax.add_patch(rect)
    ax.text(x_center, y_center, label,
            ha="center", va="center",
            fontsize=10, weight="bold", color=color,
            zorder=11)


# Define (Diamond 1 右半) 内の下三角部分に配置
add_tool_box_inside(2.5, 1.3, "ブロードリスニング", "#1a4d7a", box_w=2.0)
# Deliver (Diamond 2 右半) 内の下三角部分に配置
add_tool_box_inside(7.55, 1.3, "パブリックコメント", "#a06000")

plt.tight_layout()
out = Path("images/03_double_diamond.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"saved {out}")
