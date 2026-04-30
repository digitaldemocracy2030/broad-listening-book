"""第3章 「2つの運用フロー対照図」を生成する。

両側とも同じ6ステップ構造で並べ、政治家・行政主導型(軽量サイクル)では
ステップ3(熟議)がスキップされていることを点線+グレー塗りで明示する。
フルサイクルでは熟議後にステップ2(構造化・可視化)へ戻り再構造化されるループを示す。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "Noto Sans JP",
    "MS Gothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def draw_box(ax, x_center: float, y: float, w: float, h: float, text: str,
             fc: str = "#e3f2fd", ec: str = "#666",
             linestyle: str = "-",
             text_color: str = "#000",
             fontsize: float = 9.5, weight: str = "normal") -> None:
    rect = FancyBboxPatch(
        (x_center - w / 2, y), w, h,
        boxstyle="round,pad=0.1",
        fc=fc, ec=ec, lw=1.0, linestyle=linestyle,
    )
    ax.add_patch(rect)
    ax.text(x_center, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight)


def draw_down_arrow(ax, x: float, y_top: float, y_bot: float,
                    color: str = "#444", lw: float = 1.3,
                    linestyle: str = "-") -> None:
    ax.annotate(
        "", xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        linestyle=linestyle),
    )


fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14.5)
ax.set_aspect("auto")
ax.axis("off")

# Column titles
ax.text(2.5, 13.8, "①政治家・行政主導型", ha="center", va="center",
        fontsize=12, weight="bold", color="#1a4d7a")
ax.text(2.5, 13.2, "(軽量サイクル)", ha="center", va="center",
        fontsize=10, color="#1a4d7a")

ax.text(7.5, 13.8, "②拡張熟議支援型", ha="center", va="center",
        fontsize=12, weight="bold", color="#a06000")
ax.text(7.5, 13.2, "(フルサイクル)", ha="center", va="center",
        fontsize=10, color="#a06000")

# Box dimensions and positions (shared 7 step positions)
W = 3.4
H = 0.85
LEFT_X = 2.5
RIGHT_X = 7.5

step_y = [11.5, 10.1, 8.7, 7.3, 5.9, 4.5]

steps = [
    "1. 意見収集",
    "2. 構造化・可視化",
    "3. 熟議",
    "4. 政策立案",
    "5. 説明・フィードバック",
    "6. 再収集 (→1へ)",
]

# Light cycle (left): step 3 (熟議) is skipped
SKIPPED = {2}  # 0-indexed index for step 3

color_active_left = "#e3f2fd"
color_skipped = "#ececec"
color_active_right_common = "#e3f2fd"
color_active_right_extra = "#fff3cd"  # 熟議は強調色

# Draw left column
for i, (label, y) in enumerate(zip(steps, step_y)):
    if i in SKIPPED:
        draw_box(ax, LEFT_X, y - H / 2, W, H, label,
                 fc=color_skipped, ec="#aaa",
                 linestyle="--", text_color="#999", fontsize=9)
    else:
        draw_box(ax, LEFT_X, y - H / 2, W, H, label,
                 fc=color_active_left)

# Left column arrows (dashed for skipped portion)
for i in range(len(step_y) - 1):
    y_top = step_y[i] - H / 2
    y_bot = step_y[i + 1] + H / 2
    if i in SKIPPED or (i + 1) in SKIPPED:
        draw_down_arrow(ax, LEFT_X, y_top, y_bot,
                        color="#aaa", lw=1.0, linestyle="--")
    else:
        draw_down_arrow(ax, LEFT_X, y_top, y_bot)

# Bypass arrow on the LEFT side of left column showing 2 -> 4 直結
# (curve bulges outward to the left, away from the column)
ax.annotate(
    "",
    xy=(LEFT_X - W / 2 - 0.45, step_y[3]),  # tip at step 4(政策立案) left side
    xytext=(LEFT_X - W / 2 - 0.45, step_y[1]),  # tail at step 2 left side
    arrowprops=dict(
        arrowstyle="->",
        color="#1a4d7a", lw=1.6,
        connectionstyle="arc3,rad=0.35",
    ),
)
ax.text(LEFT_X - W / 2 - 1.4, (step_y[1] + step_y[3]) / 2,
        "市民熟議を\n経ずに直結",
        ha="right", va="center",
        fontsize=8.5, color="#1a4d7a", style="italic")

# Draw right column (all 6 steps active, step 3 highlighted as フル特有)
right_extra_set = {2}
for i, (label, y) in enumerate(zip(steps, step_y)):
    if i in right_extra_set:
        draw_box(ax, RIGHT_X, y - H / 2, W, H, label,
                 fc=color_active_right_extra, weight="bold")
    else:
        draw_box(ax, RIGHT_X, y - H / 2, W, H, label,
                 fc=color_active_right_common)

# Right column arrows (all solid)
for i in range(len(step_y) - 1):
    y_top = step_y[i] - H / 2
    y_bot = step_y[i + 1] + H / 2
    draw_down_arrow(ax, RIGHT_X, y_top, y_bot)

# Loop arrow on the RIGHT side: step 3 -> step 2 (熟議結果で再構造化)
ax.annotate(
    "",
    xy=(RIGHT_X + W / 2 + 0.4, step_y[1]),    # tip at step 2 right side (上)
    xytext=(RIGHT_X + W / 2 + 0.4, step_y[2]),  # tail at step 3 right side (下)
    arrowprops=dict(
        arrowstyle="->",
        color="#a06000", lw=1.5,
        connectionstyle="arc3,rad=0.45",
    ),
)
ax.text(RIGHT_X + W / 2 + 1.2, (step_y[1] + step_y[2]) / 2,
        "熟議結果で\n論点地図を\n再構造化",
        ha="left", va="center",
        fontsize=8.5, color="#a06000", style="italic")

# Cost annotations
ax.text(LEFT_X, 3.0,
        "コスト: 低 / 速い",
        ha="center", va="center",
        fontsize=10.5, weight="bold", color="#1a4d7a")
ax.text(LEFT_X, 2.4,
        "一人の政治家や\n小規模行政チームでも実践可",
        ha="center", va="center",
        fontsize=9, color="#444")

ax.text(RIGHT_X, 3.0,
        "コスト: 高 / 正統性が強い",
        ha="center", va="center",
        fontsize=10.5, weight="bold", color="#a06000")
ax.text(RIGHT_X, 2.4,
        "代表性のある市民熟議を経るため\n決定の正統性を補完できる",
        ha="center", va="center",
        fontsize=9, color="#444")

# Legend
ax.text(5.0, 1.0,
        "黄色のステップ(③熟議)が拡張熟議支援型の特徴。軽量サイクルではここがスキップされる",
        ha="center", va="center",
        fontsize=8.5, color="#666", style="italic")

plt.tight_layout()
out = Path("images/03_operation_flows.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"saved {out}")
