"""第11章 イスラエル/パレスチナRemesh事例の対話プロセス図(日本語版)を生成する。

論文 Konya et al., 2025 (CC BY 4.0) の Fig.1 をもとに、日本語版書籍向けに翻訳・再構成。
- 上段: 4つの対話サイクル(Uninational Phase ×3 + Joint Phase ×1)
- 下段: 各サイクル内で実行される5段階のパイプライン
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

# Color palette
COL_PHASE_UNI = "#3a5a7a"       # navy for Uninational
COL_PHASE_JOINT = "#a85b3c"     # warm rust for Joint
COL_GROUP_BG = "#dce4ef"        # very light blue
COL_GROUP_BG_JOINT = "#f4dccf"  # very light warm
COL_STEP = "#2c3e50"
COL_STEP_BG = "#fef6e4"          # warm cream
COL_TEXT = "#1a202c"
COL_LINE = "#4a5568"
COL_ARROW = "#dd6b20"


fig, ax = plt.subplots(figsize=(14.5, 9.0), dpi=200)
ax.set_xlim(0.2, 14.4)
ax.set_ylim(0, 10.0)
ax.set_aspect("auto")
ax.axis("off")


def draw_card(x_center: float, y_center: float, w: float, h: float,
              fc: str, ec: str, lw: float = 1.5, shadow: bool = True) -> None:
    if shadow:
        sh = FancyBboxPatch(
            (x_center - w / 2 + 0.05, y_center - h / 2 - 0.06), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.18",
            fc="#cbd5e0", ec="none", alpha=0.5, zorder=1,
        )
        ax.add_patch(sh)
    rect = FancyBboxPatch(
        (x_center - w / 2, y_center - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        fc=fc, ec=ec, lw=lw, zorder=2,
    )
    ax.add_patch(rect)


def draw_arrow(x1: float, y1: float, x2: float, y2: float,
               color: str = COL_ARROW, lw: float = 2.4) -> None:
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=10,head_width=7",
        color=color, lw=lw,
        shrinkA=0, shrinkB=0,
        zorder=5,
    )
    ax.add_patch(arr)


# --- Title ---
ax.text(7.3, 9.6, "イスラエル・パレスチナ平和構築者対話プロセス",
        ha="center", va="center",
        fontsize=20, weight="bold", color=COL_TEXT)
ax.text(7.3, 9.15, "4つの対話サイクル × 各サイクル内の5段階パイプライン",
        ha="center", va="center",
        fontsize=13, color="#4a5568", style="italic")

# --- Phase labels (top) ---
# Uninational Phase covers cycles 1-3
ax.plot([0.7, 9.6], [8.65, 8.65], color=COL_PHASE_UNI, lw=2.0, zorder=3)
ax.text(5.15, 8.85, "Uninational Phase（グループ別対話）",
        ha="center", va="bottom",
        fontsize=12, weight="bold", color=COL_PHASE_UNI)
# Joint Phase covers cycle 4
ax.plot([10.4, 13.9], [8.65, 8.65], color=COL_PHASE_JOINT, lw=2.0, zorder=3)
ax.text(12.15, 8.85, "Joint Phase（合同対話）",
        ha="center", va="bottom",
        fontsize=12, weight="bold", color=COL_PHASE_JOINT)

# --- 4 dialogue cycle cards ---
# Layout: 4 cards in a row
cycle_y = 7.7
cycle_w = 2.8
cycle_h = 1.5
cycle_xs = [1.7, 4.65, 7.6, 12.15]

cycles = [
    ("サイクル 1", "イスラエル系\nユダヤ人", "ヘブライ語", COL_GROUP_BG, COL_PHASE_UNI),
    ("サイクル 2", "ヨルダン川西岸地区\n・ガザのパレスチナ人", "パレスチナ・\nアラビア語", COL_GROUP_BG, COL_PHASE_UNI),
    ("サイクル 3", "イスラエル国籍の\nパレスチナ市民", "パレスチナ・\nアラビア語", COL_GROUP_BG, COL_PHASE_UNI),
    ("サイクル 4", "全グループが合同で参加", "アラビア語・\nヘブライ語・英語", COL_GROUP_BG_JOINT, COL_PHASE_JOINT),
]
for x, (cyc, group, lang, bg, ec) in zip(cycle_xs, cycles):
    draw_card(x, cycle_y, cycle_w, cycle_h, bg, ec, lw=1.8)
    ax.text(x, cycle_y + 0.45, cyc,
            ha="center", va="center",
            fontsize=11.5, weight="bold", color=ec, zorder=4)
    ax.text(x, cycle_y + 0.0, group,
            ha="center", va="center",
            fontsize=10.5, color=COL_TEXT, zorder=4)
    ax.text(x, cycle_y - 0.5, lang,
            ha="center", va="center",
            fontsize=9.5, color="#555", style="italic", zorder=4)

# Arrows between cycles
for i in range(3):
    x1 = cycle_xs[i] + cycle_w / 2 + 0.02
    x2 = cycle_xs[i + 1] - cycle_w / 2 - 0.02
    draw_arrow(x1, cycle_y, x2, cycle_y, color=COL_ARROW, lw=2.2)

# --- Connector to pipeline ---
ax.text(7.3, 6.55, "↓ 各サイクル内では下記の5段階パイプラインが実行される",
        ha="center", va="center",
        fontsize=11.5, color=COL_TEXT, weight="bold")

# --- Pipeline section background ---
panel = FancyBboxPatch(
    (0.7, 0.55), 13.2, 5.55,
    boxstyle="round,pad=0.0,rounding_size=0.2",
    fc="#fdfaf2", ec="#c8b993", lw=1.2, zorder=0,
)
ax.add_patch(panel)
ax.text(7.3, 5.85, "サイクル内のパイプライン（5段階）",
        ha="center", va="center",
        fontsize=13, weight="bold", color="#5a4a2a")

# --- 5 step boxes ---
step_y = 3.7
step_w = 2.45
step_h = 2.2
step_xs = [1.95, 4.55, 7.15, 9.75, 12.35]
steps = [
    ("1", "集団対話",
     "参加者が自由に\nテキスト入力し、\nリアルタイムで意見を共有"),
    ("2", "橋渡しステート\nメントの特定",
     "アルゴリズムが\n対立する集団の双方から\n支持される意見を抽出"),
    ("3", "集団声明への\n精緻化",
     "GPT-4 が参加者の\n言葉を保持しながら\n簡潔な声明を生成"),
    ("4", "人間専門家\nレビュー",
     "ネイティブ話者が\n文化的適切性と正確性を\n確認"),
    ("5", "参加者による\n最終投票",
     "精緻化された声明に\n参加者が投票し、\n合意度を測定"),
]
for x, (num, title, sub) in zip(step_xs, steps):
    # Card
    draw_card(x, step_y, step_w, step_h, COL_STEP_BG, COL_STEP, lw=1.5)
    # Number badge
    badge_x = x - step_w / 2 + 0.42
    badge_y = step_y + step_h / 2 - 0.42
    ax.scatter([badge_x], [badge_y], s=900,
               c=COL_STEP, edgecolors="white", linewidths=1.6, zorder=4)
    ax.text(badge_x, badge_y, num, ha="center", va="center",
            fontsize=15, weight="bold", color="white", zorder=5)
    # Title (top of card)
    ax.text(x, step_y + 0.55, title,
            ha="center", va="center",
            fontsize=12.5, weight="bold", color=COL_STEP, zorder=4)
    # Description
    ax.text(x, step_y - 0.45, sub,
            ha="center", va="center",
            fontsize=10, color=COL_TEXT, zorder=4)

# Arrows between steps
for i in range(4):
    x1 = step_xs[i] + step_w / 2 + 0.02
    x2 = step_xs[i + 1] - step_w / 2 - 0.02
    draw_arrow(x1, step_y, x2, step_y, color=COL_ARROW, lw=2.4)

# --- Source attribution ---
ax.text(7.3, 0.25,
        "出典: Konya et al., 2025 \"Using Collective Dialogues and AI to Find Common Ground Between Israeli and Palestinian Peacebuilders\" "
        "(arxiv.org/abs/2503.01769, CC BY 4.0) をもとに翻訳・再構成",
        ha="center", va="center",
        fontsize=8.5, color="#666", style="italic")

plt.tight_layout()
out = Path("images/11_04_対話プロセス_jp.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
