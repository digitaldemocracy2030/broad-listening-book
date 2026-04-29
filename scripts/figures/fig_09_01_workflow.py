"""第9章 アルティウスリンクのVOC前処理パイプライン図を生成する。

通話テキスト → 留守電除外 → ネガ判定 → 構造化抽出 → 広聴AI / BIツール の流れを示す。
本文の数字(5%除外、10〜15%に絞込)と連動。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "Noto Sans JP",
    "MS Gothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# --- Color palette (cohesive, modern) ---
COL_INPUT = "#1e3a5f"      # deep navy (input data)
COL_INPUT_BG = "#dde7f2"
COL_STEP = "#2c5282"        # blue (LLM steps)
COL_STEP_BG = "#fef3c7"     # warm pastel yellow
COL_OUTPUT = "#22543d"      # dark green
COL_OUTPUT_BG = "#c6f6d5"   # soft mint green
COL_TEXT = "#1a202c"
COL_LINE = "#4a5568"
COL_ACCENT = "#dd6b20"      # orange accent for arrows


fig, ax = plt.subplots(figsize=(15.0, 6.4), dpi=200)
ax.set_xlim(0.2, 14.4)
ax.set_ylim(0, 6.0)
ax.set_aspect("auto")
ax.axis("off")


def draw_step_box(x_center: float, y_center: float, w: float, h: float,
                  number: str, title: str, sub: str,
                  fc: str, ec: str, title_color: str = "#1a202c",
                  number_bg: str = "#2c5282") -> None:
    """番号付きの階層的なステップボックスを描画する。"""
    # Drop shadow
    shadow = FancyBboxPatch(
        (x_center - w / 2 + 0.05, y_center - h / 2 - 0.07), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        fc="#cbd5e0", ec="none", alpha=0.55, zorder=1,
    )
    ax.add_patch(shadow)
    # Main card
    rect = FancyBboxPatch(
        (x_center - w / 2, y_center - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        fc=fc, ec=ec, lw=1.6, zorder=2,
    )
    ax.add_patch(rect)
    # Number circle (top-left)
    if number:
        circ_x = x_center - w / 2 + 0.42
        circ_y = y_center + h / 2 - 0.42
        ax.scatter([circ_x], [circ_y], s=1100,
                   c=number_bg, edgecolors="white", linewidths=1.8, zorder=3)
        ax.text(circ_x, circ_y, number, ha="center", va="center",
                fontsize=19, weight="bold", color="white", zorder=4)
    # Title
    ax.text(x_center, y_center + 0.38, title,
            ha="center", va="center",
            fontsize=19, weight="bold", color=title_color, zorder=4)
    # Subtitle / detail
    ax.text(x_center, y_center - 0.45, sub,
            ha="center", va="center",
            fontsize=14.5, color="#2d3748", zorder=4)


def draw_output_box(x_center: float, y_center: float, w: float, h: float,
                    title: str, sub: str) -> None:
    shadow = FancyBboxPatch(
        (x_center - w / 2 + 0.05, y_center - h / 2 - 0.06), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        fc="#cbd5e0", ec="none", alpha=0.55, zorder=1,
    )
    ax.add_patch(shadow)
    rect = FancyBboxPatch(
        (x_center - w / 2, y_center - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        fc=COL_OUTPUT_BG, ec=COL_OUTPUT, lw=1.6, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x_center, y_center + 0.22, title,
            ha="center", va="center",
            fontsize=18, weight="bold", color=COL_OUTPUT, zorder=4)
    ax.text(x_center, y_center - 0.28, sub,
            ha="center", va="center",
            fontsize=14, color="#2d3748", zorder=4)


def draw_arrow(x1: float, y1: float, x2: float, y2: float,
               color: str = COL_ACCENT, lw: float = 2.6) -> None:
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=10,head_width=7",
        color=color, lw=lw,
        shrinkA=0, shrinkB=0,
        zorder=5,
    )
    ax.add_patch(arr)


# --- Title bar ---
ax.text(7.0, 5.55, "VOC前処理パイプライン",
        ha="center", va="center",
        fontsize=26, weight="bold", color=COL_TEXT)
ax.text(7.0, 5.0, "通話テキストから民意の構造化までの4ステップ",
        ha="center", va="center",
        fontsize=17, color="#4a5568", style="italic")

# --- Pipeline (horizontal, 4 steps) ---
y_main = 2.9
box_w = 2.7
box_h = 1.95

# Step 1: 入力
draw_step_box(
    1.7, y_main, box_w, box_h,
    number="",
    title="通話テキスト",
    sub="Speech to Text\n1日 1,000〜2,000件",
    fc=COL_INPUT_BG, ec=COL_INPUT, title_color=COL_INPUT,
)

# Step 2: 留守電除外
draw_step_box(
    4.7, y_main, box_w, box_h,
    number="1",
    title="留守電除外",
    sub="LLMで判定\n約 5% を除外",
    fc=COL_STEP_BG, ec=COL_STEP, title_color=COL_STEP,
    number_bg=COL_STEP,
)

# Step 3: ネガ判定
draw_step_box(
    7.7, y_main, box_w, box_h,
    number="2",
    title="ネガ判定",
    sub="LLMで分類\n10〜15% に絞込",
    fc=COL_STEP_BG, ec=COL_STEP, title_color=COL_STEP,
    number_bg=COL_STEP,
)

# Step 4: 構造化抽出
draw_step_box(
    10.7, y_main, box_w, box_h,
    number="3",
    title="構造化抽出",
    sub="Claude Structured\nVOC / コンタクトリーズン\n+ メタ情報5〜6項目",
    fc=COL_STEP_BG, ec=COL_STEP, title_color=COL_STEP,
    number_bg=COL_STEP,
)

# --- Outputs (branch from step 4) ---
draw_output_box(13.0, 4.55, 2.2, 1.15,
                "広聴AI", "クラスタ可視化")
draw_output_box(13.0, 1.25, 2.2, 1.15,
                "BIツール", "定点観測")

# --- Arrows ---
# Pipeline horizontal
draw_arrow(3.05, y_main, 3.35, y_main)
draw_arrow(6.05, y_main, 6.35, y_main)
draw_arrow(9.05, y_main, 9.35, y_main)
# Branch from step 4 → outputs
draw_arrow(11.95, 3.55, 12.3, 4.15)   # to 広聴AI
draw_arrow(11.95, 2.25, 12.3, 1.65)   # to BIツール

# --- Stage labels (under each step) ---
ax.text(1.7, 1.55, "INPUT",
        ha="center", va="center",
        fontsize=15, weight="bold", color=COL_INPUT, alpha=0.85)
for x in (4.7, 7.7):
    ax.text(x, 1.55, "FILTER",
            ha="center", va="center",
            fontsize=15, weight="bold", color=COL_STEP, alpha=0.85)
ax.text(10.7, 1.55, "EXTRACT",
        ha="center", va="center",
        fontsize=15, weight="bold", color=COL_STEP, alpha=0.85)
ax.text(13.0, 2.95, "OUTPUT",
        ha="center", va="center",
        fontsize=15, weight="bold", color=COL_OUTPUT, alpha=0.85)

plt.tight_layout()
out = Path("images/09_01_workflow.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
