"""
公明党 We Connect の政策立案フロー図を生成するスクリプト。

出典: 公明党Webサイト「公明ハンドブック2025 対話の強い味方、活用しよう！」
      https://www.komei.or.jp/komeinews/p406695/
の図を参考に、本書向けに自前で再描画したもの。

使い方:
    uv run python scripts/weconnect_workflow_figure.py

出力: images/06_05_WeConnect_workflow.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Yu Gothic",
    "Meiryo",
    "Hiragino Sans",
    "MS Gothic",
    "sans-serif",
]

OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


# (時期, ステップ名, フェーズ)
# フェーズで色分け: round1=第一弾サイクル, round2=第二弾サイクル, decide=決定・発表
STEPS = [
    ("3月17日〜", "アンケート第一弾開始", "round1"),
    ("", "寄せられた声を集約", "round1"),
    ("4月中下旬〜", "党内で議論", "round1"),
    ("5月GW明け", "アンケート第二弾\n(複数の政策案提示)", "round2"),
    ("", "寄せられた意見を基に\n政策案をさらに議論", "round2"),
    ("6月", "党政務調査会で政策決定", "decide"),
    ("参院選公示前", "参院選重点政策に反映\n政策発表", "decide"),
]

PHASE_COLORS = {
    "round1": "#cfe3ff",
    "round2": "#cfe9d6",
    "decide": "#f5d6cf",
}
PHASE_EDGES = {
    "round1": "#3a6fb0",
    "round2": "#3a8a55",
    "decide": "#b06a3a",
}

SOURCE_NOTE = "出典: 公明党Webサイト (komei.or.jp/komeinews/p406695) の図を基に作成"


def draw_workflow():
    n = len(STEPS)
    # box: 200pt 幅減・60pt 高さ減 (font 40pt 換算で約 2.78in / 0.83in)
    box_w = 11.5 - 200 / 72
    box_h = 2.8 - 60 / 72
    gap = 1.2
    when_pad = 5.0

    fig_w = box_w + when_pad + 1.4
    fig_h = n * (box_h + gap) + 3.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_box_left = when_pad
    x_box_right = when_pad + box_w
    x_box_center = when_pad + box_w / 2

    # 上から下へ並べる
    y_top = n * (box_h + gap) - gap

    for i, (when, label, phase) in enumerate(STEPS):
        y_center = y_top - i * (box_h + gap) - box_h / 2
        # 時期ラベル（左側）
        if when:
            ax.text(
                when_pad - 0.3,
                y_center,
                when,
                ha="right",
                va="center",
                fontsize=38,
                color="#333333",
            )
        # ボックス
        box = FancyBboxPatch(
            (x_box_left, y_center - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.8,
            facecolor=PHASE_COLORS[phase],
            edgecolor=PHASE_EDGES[phase],
        )
        ax.add_patch(box)
        ax.text(
            x_box_center,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=38,
            color="#1a1a1a",
        )
        # 矢印（次のステップへ）
        if i < n - 1:
            y_next_top = y_center - box_h / 2
            y_next_bottom = y_next_top - gap + 0.1
            arrow = FancyArrowPatch(
                (x_box_center, y_next_top - 0.02),
                (x_box_center, y_next_bottom),
                arrowstyle="-|>",
                mutation_scale=22,
                linewidth=2.0,
                color="#555555",
            )
            ax.add_patch(arrow)

    # タイトル
    ax.text(
        x_box_center,
        y_top + 1.6,
        "政策立案までの流れ",
        ha="center",
        va="bottom",
        fontsize=44,
        fontweight="bold",
        color="#1a1a1a",
    )

    # 出典
    ax.text(
        x_box_center,
        -1.2,
        SOURCE_NOTE,
        ha="center",
        va="top",
        fontsize=22,
        color="#666666",
    )

    ax.set_xlim(-0.2, x_box_right + 0.4)
    ax.set_ylim(-2.8, y_top + 3.2)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_05_WeConnect_workflow.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    draw_workflow()
