"""
クラスタリング解説用の図を生成するスクリプト

生成される図:
1. 2つのガウス分布（シンプルなクラスタリング例）
2. 複雑なデータ構造に対する各アルゴリズムの比較
3. k-meansのステップバイステップ解説
4. Ward法（階層的クラスタリング）の解説とデンドログラム

使い方:
    uv run python scripts/clustering_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# 日本語フォント設定
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']

# 出力ディレクトリ
OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

# 乱数シード固定
np.random.seed(42)


def generate_two_gaussians(n_samples=100, max_dist=1.2):
    """2つのガウス分布からデータを生成（外れ値を除去）"""
    centers = [np.array([-2, 2]), np.array([2, -2])]
    std = 0.5

    clusters = []
    for center in centers:
        points = []
        while len(points) < n_samples:
            # ガウス分布から点を生成
            point = np.random.randn(2) * std + center
            # 中心からの距離がmax_dist以内なら採用
            if np.linalg.norm(point - center) <= max_dist:
                points.append(point)
        clusters.append(np.array(points))

    return np.vstack(clusters), np.array([0]*n_samples + [1]*n_samples)


def fig1_two_gaussians():
    """図1: 2つのガウス分布（クラスタリング前）"""
    np.random.seed(42)  # 図1と図2で同じデータを使うためシードを固定
    X, _ = generate_two_gaussians()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('データの分布')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_clustering_two_gaussians.png", dpi=150)
    plt.close()
    print(f"Saved: 13_clustering_two_gaussians.png")


def fig2_two_gaussians_clustered():
    """図2: 2つのガウス分布（クラスタリング後、丸で囲む）"""
    np.random.seed(42)  # 図1と図2で同じデータを使うためシードを固定
    X, labels = generate_two_gaussians()

    fig, ax = plt.subplots(figsize=(8, 6))

    # 点をプロット（カラー + 形状で区別）
    styles = [
        {'color': '#E74C3C', 'marker': 'o'},  # 赤丸
        {'color': '#3498DB', 'marker': 's'},  # 青四角
    ]
    for i in range(2):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=styles[i]['color'], marker=styles[i]['marker'],
                   alpha=0.6, s=50, label=f'グループ {i+1}')

    # 円で囲む（データ生成時のmax_distに合わせる）
    from matplotlib.patches import Circle
    centers = [np.array([-2, 2]), np.array([2, -2])]
    radius = 1.3  # max_dist + 余裕
    for i in range(2):
        circle = Circle(centers[i], radius,
                        fill=False, edgecolor=styles[i]['color'], linewidth=2, linestyle='--')
        ax.add_patch(circle)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('自然なグループ分け')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_clustering_two_gaussians_clustered.png", dpi=150)
    plt.close()
    print(f"Saved: 13_clustering_two_gaussians_clustered.png")


def fig3_complex_datasets():
    """図3: 複雑なデータ構造と各アルゴリズムの比較（scikit-learn風）"""
    from sklearn.datasets import make_circles, make_moons, make_blobs
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

    np.random.seed(42)

    # データセット生成
    n_samples = 300

    # 1つの塊（単一のガウス分布）
    single_blob = np.random.randn(n_samples, 2) * 0.5

    datasets = [
        ("1つの塊", single_blob),
        ("円形", make_circles(n_samples=n_samples, factor=0.5, noise=0.05)[0]),
        ("三日月", make_moons(n_samples=n_samples, noise=0.05)[0]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # カラー + 形状で区別（グレースケールでも判別可能に）
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    algo_names = ["K-means", "Ward法", "DBSCAN"]

    for i, (data_name, X) in enumerate(datasets):
        # データを正規化
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        for j, algo_name in enumerate(algo_names):
            ax = axes[i, j]

            # クラスタリング実行（データパターンごとに適切なパラメータ）
            if data_name == "1つの塊":
                # K-meansとWard法は無理やり2つに分ける
                if algo_name == "K-means":
                    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
                elif algo_name == "Ward法":
                    labels = AgglomerativeClustering(n_clusters=2).fit_predict(X)
                else:  # DBSCAN
                    labels = DBSCAN(eps=0.3).fit_predict(X)
            elif data_name == "3つの塊":
                # 3つの塊を2つに分けろと指示 → 2つが無理やり1つにまとめられる
                if algo_name == "K-means":
                    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
                elif algo_name == "Ward法":
                    labels = AgglomerativeClustering(n_clusters=2).fit_predict(X)
                else:  # DBSCAN
                    labels = DBSCAN(eps=0.5).fit_predict(X)
            else:  # 円形、三日月
                if algo_name == "K-means":
                    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
                elif algo_name == "Ward法":
                    labels = AgglomerativeClustering(n_clusters=2).fit_predict(X)
                else:  # DBSCAN
                    labels = DBSCAN(eps=0.3).fit_predict(X)

            # プロット（カラー + 形状で区別）
            unique_labels = sorted(set(labels))
            for k in unique_labels:
                mask = labels == k
                if k == -1:  # ノイズ（DBSCAN）
                    ax.scatter(X[mask, 0], X[mask, 1], c='gray', alpha=0.3, s=20, marker='x')
                else:
                    ax.scatter(X[mask, 0], X[mask, 1], c=colors[k % len(colors)],
                               marker=markers[k % len(markers)], alpha=0.6, s=25)

            if i == 0:
                ax.set_title(algo_name, fontsize=14)
            if j == 0:
                ax.set_ylabel(data_name, fontsize=14)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('データ構造とアルゴリズムによるクラスタリング結果の違い', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_clustering_algorithms_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: 13_clustering_algorithms_comparison.png")


def fig4_kmeans_steps():
    """図4: k-meansのステップバイステップ解説（割り当てと中心移動を分離）"""
    X, true_labels = generate_two_gaussians(n_samples=50)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # カラー + 形状で区別
    styles = [
        {'color': '#E74C3C', 'marker': 'o'},  # 赤丸
        {'color': '#3498DB', 'marker': 's'},  # 青四角
    ]

    # ステップ0: 初期データ
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50)
    ax.set_title('Step 0: 初期データ', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 初期中心をランダムに設定（見やすい位置に固定）
    centers = np.array([[0, 0], [1, 1]])

    # ステップ1: 中心点を配置
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50)
    for i in range(2):
        ax.scatter(centers[i, 0], centers[i, 1], c=styles[i]['color'], marker='*',
                   s=400, edgecolors='black', linewidths=1)
    ax.set_title('Step 1: 中心点を配置', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # k-meansを手動で実行（割り当てと中心移動を分離）
    step_idx = 2
    for iteration in range(3):
        # 各点を最も近い中心に割り当て
        distances = np.sqrt(((X[:, np.newaxis] - centers) ** 2).sum(axis=2))
        labels = distances.argmin(axis=1)

        # 割り当てステップを表示
        ax = axes[step_idx]
        for i in range(2):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], c=styles[i]['color'],
                       marker=styles[i]['marker'], alpha=0.6, s=50)
        # 現在の中心点を表示
        for i in range(2):
            ax.scatter(centers[i, 0], centers[i, 1], c=styles[i]['color'], marker='*',
                       s=400, edgecolors='black', linewidths=1)
        ax.set_title(f'Step {step_idx}: 割り当て', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        step_idx += 1

        # 新しい中心を計算
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(2)])

        # 中心移動ステップを表示（最後の反復以外）
        if step_idx < len(axes):
            ax = axes[step_idx]
            for i in range(2):
                mask = labels == i
                ax.scatter(X[mask, 0], X[mask, 1], c=styles[i]['color'],
                           marker=styles[i]['marker'], alpha=0.6, s=50)
            # 古い中心（薄く）
            for i in range(2):
                ax.scatter(centers[i, 0], centers[i, 1], c=styles[i]['color'], marker='*',
                           s=400, edgecolors='black', linewidths=1, alpha=0.3)
            # 新しい中心
            for i in range(2):
                ax.scatter(new_centers[i, 0], new_centers[i, 1], c=styles[i]['color'], marker='*',
                           s=400, edgecolors='black', linewidths=1)
            # 矢印で移動を示す
            for i in range(2):
                ax.annotate('', xy=new_centers[i], xytext=centers[i],
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            ax.set_title(f'Step {step_idx}: 中心移動', fontsize=11)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            step_idx += 1

        centers = new_centers

    plt.suptitle('K-meansアルゴリズムの動作', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_kmeans_steps.png", dpi=150)
    plt.close()
    print(f"Saved: 13_kmeans_steps.png")


def fig5_hierarchical_dendrogram():
    """図5: 階層的クラスタリングとデンドログラム"""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    # シンプルな8点のデータ
    np.random.seed(123)
    X = np.array([
        [-3, 0], [-2.5, 0.5],  # グループA
        [-1, 0], [-0.5, 0.3],  # グループB
        [1, 0], [1.5, 0.2],    # グループC
        [3, 0], [3.5, 0.4],    # グループD
    ])
    labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 2次元プロット（カラー + 形状で区別）
    ax = axes[0]
    group_styles = [
        {'color': '#E74C3C', 'marker': 'o'},  # A: 赤丸
        {'color': '#3498DB', 'marker': 's'},  # B: 青四角
        {'color': '#2ECC71', 'marker': '^'},  # C: 緑三角
        {'color': '#9B59B6', 'marker': 'D'},  # D: 紫ダイヤ
    ]
    for i, (x, label) in enumerate(zip(X, labels)):
        group_idx = i // 2
        ax.scatter(x[0], x[1], c=group_styles[group_idx]['color'],
                   marker=group_styles[group_idx]['marker'], s=100)
        ax.annotate(label, (x[0], x[1]), xytext=(5, 5),
                    textcoords='offset points', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('データ点の分布', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 右: デンドログラム
    ax = axes[1]
    Z = linkage(X, method='ward')
    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=12)
    ax.set_ylabel('距離')
    ax.set_title('デンドログラム（Ward法）', fontsize=14)

    # 切断線を追加
    ax.axhline(y=3, color='red', linestyle='--', linewidth=2, label='2グループに分割')
    ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, label='4グループに分割')
    ax.legend(loc='upper right')

    plt.suptitle('階層的クラスタリング', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_hierarchical_dendrogram.png", dpi=150)
    plt.close()
    print(f"Saved: 13_hierarchical_dendrogram.png")


def fig6_ward_steps():
    """図6: Ward法のステップバイステップ解説"""
    # シンプルな6点のデータ
    X = np.array([
        [-2, 0], [-1.5, 0.3],  # 近い2点
        [0, 0], [0.5, 0.2],    # 近い2点
        [2, 0], [2.3, 0.3],    # 近い2点
    ])
    labels = ['A', 'B', 'C', 'D', 'E', 'F']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors_list = [
        ['gray'] * 6,  # Step 0
        ['#E74C3C', '#E74C3C', 'gray', 'gray', 'gray', 'gray'],  # Step 1: A-Bが結合
        ['#E74C3C', '#E74C3C', '#3498DB', '#3498DB', 'gray', 'gray'],  # Step 2: C-Dが結合
        ['#E74C3C', '#E74C3C', '#3498DB', '#3498DB', '#2ECC71', '#2ECC71'],  # Step 3: E-Fが結合
        ['#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#2ECC71', '#2ECC71'],  # Step 4: AB-CDが結合
        ['#9B59B6'] * 6,  # Step 5: 全部結合
    ]

    titles = [
        'Step 0: 初期状態（全員バラバラ）',
        'Step 1: 最も近いA-Bを結合',
        'Step 2: 次に近いC-Dを結合',
        'Step 3: 次に近いE-Fを結合',
        'Step 4: (AB)と(CD)を結合',
        'Step 5: 全て結合',
    ]

    for step, (ax, colors, title) in enumerate(zip(axes, colors_list, titles)):
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=150, edgecolors='black', linewidths=1)
        for i, label in enumerate(labels):
            ax.annotate(label, (X[i, 0], X[i, 1]), xytext=(5, 8),
                        textcoords='offset points', fontsize=14, fontweight='bold')

        # 結合を線で示す
        if step >= 1:
            ax.plot([X[0, 0], X[1, 0]], [X[0, 1], X[1, 1]], 'k-', linewidth=2)
        if step >= 2:
            ax.plot([X[2, 0], X[3, 0]], [X[2, 1], X[3, 1]], 'k-', linewidth=2)
        if step >= 3:
            ax.plot([X[4, 0], X[5, 0]], [X[4, 1], X[5, 1]], 'k-', linewidth=2)
        if step >= 4:
            # ABとCDの中心を結ぶ
            center_ab = X[0:2].mean(axis=0)
            center_cd = X[2:4].mean(axis=0)
            ax.plot([center_ab[0], center_cd[0]], [center_ab[1], center_cd[1]], 'k--', linewidth=2)
        if step >= 5:
            # ABCDとEFの中心を結ぶ
            center_abcd = X[0:4].mean(axis=0)
            center_ef = X[4:6].mean(axis=0)
            ax.plot([center_abcd[0], center_ef[0]], [center_abcd[1], center_ef[1]], 'k--', linewidth=2)

        ax.set_title(title, fontsize=12)
        ax.set_xlim(-3, 3.5)
        ax.set_ylim(-1, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Ward法（階層的クラスタリング）の動作', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_ward_steps.png", dpi=150)
    plt.close()
    print(f"Saved: 13_ward_steps.png")


def main():
    """全ての図を生成"""
    print("クラスタリング解説用の図を生成します...")
    print(f"出力先: {OUTPUT_DIR}")
    print()

    fig1_two_gaussians()
    fig2_two_gaussians_clustered()
    fig3_complex_datasets()
    fig4_kmeans_steps()
    fig5_hierarchical_dendrogram()
    fig6_ward_steps()

    print()
    print("完了しました！")


if __name__ == "__main__":
    main()
