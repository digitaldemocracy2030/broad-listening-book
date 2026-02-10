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

    # 3つの塊
    three_blobs = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.5, random_state=42)[0]

    datasets = [
        ("1つの塊", single_blob),
        ("円形", make_circles(n_samples=n_samples, factor=0.5, noise=0.05)[0]),
        ("三日月", make_moons(n_samples=n_samples, noise=0.05)[0]),
        ("3つの塊", three_blobs),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

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


def fig_mnist_samples():
    """MNISTデータセットのサンプル画像"""
    from sklearn.datasets import fetch_openml

    print("MNISTサンプル画像を生成中...")
    # MNISTデータセットを読み込み
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # 各数字から3枚ずつサンプルを選択
    n_samples_per_digit = 3
    fig, axes = plt.subplots(n_samples_per_digit, 10, figsize=(12, 4))

    np.random.seed(42)
    for digit in range(10):
        # その数字のインデックスを取得
        digit_indices = np.where(y == digit)[0]
        # ランダムに3枚選択
        selected = np.random.choice(digit_indices, n_samples_per_digit, replace=False)

        for row, idx in enumerate(selected):
            ax = axes[row, digit]
            # 28x28の画像として表示
            ax.imshow(X[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(str(digit), fontsize=14, fontweight='bold')

    plt.suptitle('MNIST手書き数字データセット（28×28ピクセル = 784次元）', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_mnist_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 13_mnist_samples.png")


def fig_mnist_ambiguous():
    """MNISTの分類困難なサンプル（分類器が誤分類した画像）"""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb

    print("MNIST分類困難サンプルを生成中...")
    # MNISTデータセットを読み込み
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # LightGBMで分類器を訓練（非線形モデルで相互作用を学習）
    print("LightGBM分類器を訓練中...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # テストデータで予測
    y_pred = clf.predict(X_test)

    # 誤分類されたサンプルを収集
    misclassified_mask = y_pred != y_test
    misclassified_indices = np.where(misclassified_mask)[0]
    print(f"誤分類数: {len(misclassified_indices)} / {len(y_test)}")

    # 予測確率を取得
    y_proba = clf.predict_proba(X_test)

    # 表示したいペア（正解ラベル, 予測ラベル）
    desired_pairs = [
        (6, 0),  # 6を0と誤認
        (0, 6),  # 0を6と誤認
        (5, 3),  # 5を3と誤認
        (3, 5),  # 3を5と誤認
        (9, 4),  # 9を4と誤認
        (4, 9),  # 4を9と誤認
        (2, 7),  # 2を7と誤認
        (8, 3),  # 8を3と誤認
    ]

    # 各ペアについて、正解ラベルの確率が最も低い（最も自信を持って間違えた）サンプルを探す
    misclassified_samples = []
    for true_label, pred_label in desired_pairs:
        # このペアに該当する誤分類を探す
        candidates = []
        for idx in misclassified_indices:
            if y_test[idx] == true_label and y_pred[idx] == pred_label:
                # 正解ラベルの確率（低いほど「自信を持って間違えた」）
                true_label_prob = y_proba[idx, true_label]
                candidates.append((idx, true_label_prob))

        if candidates:
            # 正解ラベルの確率が最も低いものを選択
            best_idx = min(candidates, key=lambda x: x[1])[0]
            misclassified_samples.append({
                'image': X_test[best_idx],
                'true_label': true_label,
                'pred_label': pred_label,
            })
        else:
            # 見つからない場合は、true_labelの誤分類で最も自信のあるものを選ぶ
            candidates = []
            for idx in misclassified_indices:
                if y_test[idx] == true_label:
                    true_label_prob = y_proba[idx, true_label]
                    candidates.append((idx, true_label_prob, y_pred[idx]))

            if candidates:
                best = min(candidates, key=lambda x: x[1])
                misclassified_samples.append({
                    'image': X_test[best[0]],
                    'true_label': true_label,
                    'pred_label': best[2],
                })
            else:
                print(f"Warning: {true_label}の誤分類が見つかりません")

    # 可視化（2行4列）
    n_samples = min(8, len(misclassified_samples))
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    axes = axes.flatten()

    for i in range(n_samples):
        sample = misclassified_samples[i]
        ax = axes[i]
        ax.imshow(sample['image'].reshape(28, 28), cmap='gray')
        ax.set_title(f"正解: {sample['true_label']}\nAIの予測: {sample['pred_label']}",
                     fontsize=11)
        ax.axis('off')

    # 余ったaxesを非表示
    for i in range(n_samples, 8):
        axes[i].axis('off')

    plt.suptitle('AIが誤分類した手書き数字の例', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(OUTPUT_DIR / "13_mnist_ambiguous.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 13_mnist_ambiguous.png")


def fig_mnist_boundary_zoom():
    """MNISTのUMAP境界付近を拡大し、実際の画像を表示"""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import umap

    print("MNIST境界拡大図を生成中...")
    # MNISTデータセットを読み込み
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # サンプル数を削減してUMAP実行
    n_samples = 10000
    np.random.seed(42)
    random_idx = np.random.choice(X.shape[0], n_samples, replace=False)
    X_subset = X[random_idx]
    y_subset = y[random_idx]

    # 標準化してUMAP実行
    X_scaled = StandardScaler().fit_transform(X_subset)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 指定範囲でフィルタリング（4, 9が混在する領域）
    # 重心: 4=(1.20, 0.93), 9=(1.05, 2.34)
    umap1_min, umap1_max = 0.0, 2.5
    umap2_min, umap2_max = 0.0, 3.5

    mask = ((X_umap[:, 0] >= umap1_min) & (X_umap[:, 0] <= umap1_max) &
            (X_umap[:, 1] >= umap2_min) & (X_umap[:, 1] <= umap2_max))

    # 4と9のみに絞る
    label_mask = (y_subset == 4) | (y_subset == 9)
    mask = mask & label_mask

    X_filtered = X_subset[mask]
    y_filtered = y_subset[mask]
    coords_filtered = X_umap[mask]

    print(f"範囲内のサンプル数: {len(X_filtered)}")

    # サンプルが多すぎる場合は間引く
    max_samples = 60
    if len(X_filtered) > max_samples:
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_filtered), max_samples, replace=False)
        X_filtered = X_filtered[sample_idx]
        y_filtered = y_filtered[sample_idx]
        coords_filtered = coords_filtered[sample_idx]
        print(f"間引き後のサンプル数: {len(X_filtered)}")

    # 可視化
    fig, ax = plt.subplots(figsize=(12, 10))

    # ラベルごとの枠の色
    label_colors = {
        4: '#e74c3c',  # 赤
        9: '#3498db',  # 青
    }

    # 各点に画像を配置
    for i in range(len(X_filtered)):
        img = X_filtered[i].reshape(28, 28)
        label = y_filtered[i]
        edge_color = label_colors.get(label, 'gray')
        # 画像を表示
        imagebox = OffsetImage(img, zoom=0.8, cmap='gray')
        ab = AnnotationBbox(imagebox, (coords_filtered[i, 0], coords_filtered[i, 1]),
                           frameon=True, pad=0.1,
                           bboxprops=dict(edgecolor=edge_color, linewidth=2))
        ax.add_artist(ab)

    # 軸の設定
    ax.set_xlim(umap1_min - 0.2, umap1_max + 0.2)
    ax.set_ylim(umap2_min - 0.1, umap2_max + 0.1)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'UMAP境界付近の拡大（4と9が混在する領域）\nUMAP1: [{umap1_min}, {umap1_max}], UMAP2: [{umap2_min}, {umap2_max}]',
                 fontsize=14)
    ax.grid(True, alpha=0.3)

    # 凡例用のダミープロット（枠の色で区別）
    for digit, color in sorted(label_colors.items()):
        ax.scatter([], [], label=f'{digit}', s=200, marker='s',
                   facecolors='white', edgecolors=color, linewidths=3)
    ax.legend(title='ラベル（枠の色）', loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_mnist_boundary_zoom.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 13_mnist_boundary_zoom.png")


def fig7_pca_vs_umap():
    """図7: PCAとUMAPの比較（MNISTデータセット）"""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import umap

    print("MNISTデータセットを読み込み中...")
    # MNISTデータセットを読み込み（784次元の手書き数字データ）
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    # データが大きいのでサンプル数を削減（5000サンプル使用）
    n_samples = 5000
    random_idx = np.random.RandomState(42).choice(X.shape[0], n_samples, replace=False)
    X = X[random_idx]
    y = y[random_idx].astype(int)
    print(f"データ形状: {X.shape}")

    # データを標準化
    print("データを標準化中...")
    X_scaled = StandardScaler().fit_transform(X)

    # PCAで2次元に圧縮
    print("PCAで2次元に圧縮中...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # UMAPで2次元に圧縮
    print("UMAPで2次元に圧縮中...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 可視化
    print("図を作成中...")
    fig = plt.figure(figsize=(14, 6))

    # カラーマップとマーカー形状
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'P']  # 10種類のマーカー

    # PCAによる可視化
    ax1 = fig.add_subplot(121)
    for i in range(10):
        ax1.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    color=colors[i], marker=markers[i], alpha=0.5, s=15, label=str(i))
    ax1.set_title('PCAによる2次元圧縮\n（数字が混ざり合っている）', fontsize=14)
    ax1.set_xlabel('第1主成分')
    ax1.set_ylabel('第2主成分')

    # UMAPによる可視化
    ax2 = fig.add_subplot(122)
    for i in range(10):
        ax2.scatter(X_umap[y == i, 0], X_umap[y == i, 1],
                    color=colors[i], marker=markers[i], alpha=0.5, s=15, label=str(i))

    # 各数字の重心にラベルを配置
    for i in range(10):
        centroid_x = X_umap[y == i, 0].mean()
        centroid_y = X_umap[y == i, 1].mean()
        ax2.annotate(str(i), (centroid_x, centroid_y),
                     fontsize=14, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                               edgecolor='black', linewidth=1.5, alpha=0.9))

    ax2.set_title('UMAPによる2次元圧縮\n（数字ごとに分離されている）', fontsize=14)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')

    # 凡例を下部に配置
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=10,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)

    plt.suptitle('次元圧縮手法の比較：MNIST手書き数字（784次元→2次元）', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "13_pca_vs_umap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 13_pca_vs_umap.png")


def fig8_pca_2d_to_1d():
    """図8: PCAによる2次元から1次元への次元削減デモ"""
    np.random.seed(42)

    # 斜めに伸びた楕円形のデータを生成（主成分が斜め方向になる）
    n_samples = 80
    # 主成分方向（斜め45度より少し傾いた方向）
    angle = np.pi / 4  # 45度
    # 主成分方向の分散を大きく、直交方向の分散を小さく
    std_main = 2.0  # 主成分方向の標準偏差
    std_minor = 0.5  # 直交方向の標準偏差

    # 主成分方向のベクトル
    pc1 = np.array([np.cos(angle), np.sin(angle)])
    pc2 = np.array([-np.sin(angle), np.cos(angle)])

    # データ生成
    t_main = np.random.randn(n_samples) * std_main
    t_minor = np.random.randn(n_samples) * std_minor
    X = np.outer(t_main, pc1) + np.outer(t_minor, pc2)

    # PCAを実行
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    X_projected = pca.fit_transform(X)

    # 主成分方向
    pc_direction = pca.components_[0]
    mean = pca.mean_

    # 投影点を計算（2D空間上での位置）
    projected_2d = np.outer(X_projected.flatten(), pc_direction) + mean

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 左: 元の2次元データ
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c='#3498DB', alpha=0.6, s=50)
    ax.set_xlabel('変数1')
    ax.set_ylabel('変数2')
    ax.set_title('元の2次元データ', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # 中央: 主成分軸と投影
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c='#3498DB', alpha=0.4, s=50, label='元データ')

    # 主成分軸を描画（十分長い線）
    line_length = 5
    line_start = mean - pc_direction * line_length
    line_end = mean + pc_direction * line_length
    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]],
            'r-', linewidth=3, label='第1主成分軸')

    # 投影線（元の点から投影点への線）
    for i in range(n_samples):
        ax.plot([X[i, 0], projected_2d[i, 0]], [X[i, 1], projected_2d[i, 1]],
                'gray', alpha=0.3, linewidth=0.5)

    # 投影点
    ax.scatter(projected_2d[:, 0], projected_2d[:, 1], c='#E74C3C', alpha=0.8, s=30, marker='s', label='投影点')

    ax.set_xlabel('変数1')
    ax.set_ylabel('変数2')
    ax.set_title('主成分軸への投影', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend(loc='upper left')

    # 右: 1次元に圧縮されたデータ
    ax = axes[2]
    # 1次元データを横軸に表示（y=0の線上に）
    ax.scatter(X_projected.flatten(), np.zeros(n_samples), c='#E74C3C', alpha=0.8, s=50, marker='s')
    ax.axhline(y=0, color='red', linewidth=2)
    ax.set_xlabel('第1主成分')
    ax.set_yticks([])
    ax.set_title('1次元に圧縮されたデータ', fontsize=14)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('PCAによる次元削減（2次元 → 1次元）', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_pca_2d_to_1d.png", dpi=150)
    plt.close()
    print(f"Saved: 13_pca_2d_to_1d.png")


def fig_word2vec_umap():
    """word2vecベクトルのUMAP可視化（動物・地名・料理）"""
    from gensim.models import KeyedVectors
    from huggingface_hub import hf_hub_download
    import umap

    print("word2vec + UMAP可視化を生成中...")

    # 日本語Wikipedia学習済みWord2Vecモデルを読み込み
    print("日本語Word2Vecモデルを読み込み中...")
    model = KeyedVectors.load_word2vec_format(
        hf_hub_download(repo_id="Word2vec/wikipedia2vec_jawiki_20180420_300d",
                        filename="jawiki_20180420_300d.txt")
    )

    # 単語リスト（日本語）
    animals = [
        "犬", "猫", "ライオン", "トラ",
        "ゾウ", "馬", "牛", "豚",
        "羊", "ウサギ", "クマ", "オオカミ",
        "キツネ", "シカ", "サル", "イルカ",
        "クジラ", "サメ", "ワシ", "フクロウ",
    ]

    places = [
        "東京", "ロンドン", "パリ", "北京",
        "ソウル", "シドニー", "ローマ", "ベルリン",
        "モスクワ", "カイロ", "バンコク", "シンガポール",
        "ドバイ", "アムステルダム", "ストックホルム",
        "オスロ", "ウィーン", "マドリード",
        "リスボン", "アテネ",
    ]

    foods = [
        "ピザ", "寿司", "パスタ", "カレー",
        "ハンバーガー", "ステーキ", "サラダ", "スープ",
        "パン", "ご飯", "麺", "サンドイッチ",
        "タコス", "ラーメン", "餃子", "ケーキ",
        "パイ", "チョコレート", "チーズ", "ワイン",
    ]

    sports = [
        "サッカー", "野球", "テニス", "水泳",
        "バスケットボール", "バレーボール", "卓球", "柔道",
        "スキー", "ゴルフ", "ボクシング", "マラソン",
        "ラグビー", "バドミントン", "体操", "レスリング",
        "フェンシング", "アーチェリー", "カヌー", "スケート",
    ]

    # ベクトルを取得
    words = []
    labels_jp = []
    categories = []
    vectors = []

    for cat_name, word_list in [("動物", animals), ("地名", places), ("料理", foods), ("スポーツ", sports)]:
        for word in word_list:
            if word in model:
                words.append(word)
                labels_jp.append(word)
                categories.append(cat_name)
                vectors.append(model[word])
            else:
                print(f"  語彙になし: {word}")

    vectors = np.array(vectors)
    print(f"取得した単語数: {len(vectors)}, ベクトル次元: {vectors.shape[1]}")

    # UMAPで2次元に圧縮
    print("UMAPで次元圧縮中...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.9, spread=3.0, random_state=42)
    X_umap = reducer.fit_transform(vectors)

    # 可視化
    fig, ax = plt.subplots(figsize=(16, 12))

    # カテゴリごとの色とマーカー
    category_styles = {
        "動物": {"color": "#E74C3C", "marker": "o"},
        "地名": {"color": "#3498DB", "marker": "s"},
        "料理": {"color": "#2ECC71", "marker": "^"},
        "スポーツ": {"color": "#9B59B6", "marker": "D"},
    }

    # カテゴリごとにプロット
    for cat, style in category_styles.items():
        mask = [c == cat for c in categories]
        indices = [i for i, m in enumerate(mask) if m]
        ax.scatter(
            X_umap[indices, 0], X_umap[indices, 1],
            c=style["color"], marker=style["marker"],
            s=120, alpha=0.8, edgecolors='black', linewidths=0.5,
            label=cat
        )

    # 各点にラベルを追加
    for i, label in enumerate(labels_jp):
        ax.annotate(
            label,
            (X_umap[i, 0], X_umap[i, 1]),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=18,
            alpha=0.9
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Word2VecベクトルのUMAP可視化\n（300次元 → 2次元）', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_word2vec_umap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 13_word2vec_umap.png")


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
    fig_mnist_samples()
    fig_mnist_ambiguous()
    fig_mnist_boundary_zoom()
    fig7_pca_vs_umap()
    fig8_pca_2d_to_1d()
    fig_word2vec_umap()

    print()
    print("完了しました！")


if __name__ == "__main__":
    main()
