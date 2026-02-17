#!/usr/bin/env python3
"""
都道府県データを使ったUMAP次元圧縮の例

「統計でみる都道府県のすがた2024」のデータを使用して、
47都道府県を多次元の統計指標からUMAPで2次元に圧縮し可視化します。

データ出典: 総務省統計局「統計でみる都道府県のすがた2024」
https://www.e-stat.go.jp/stat-search/files?toukei=00200502&tstat=000001213120
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import urllib.request
import os

# 日本語フォント設定
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP']

# 出力ディレクトリ
OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

# データディレクトリ
DATA_DIR = Path(__file__).parent.parent / "data" / "estat"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 都道府県名（英語→日本語の対応表）
PREFECTURE_EN_TO_JP = {
    "Hokkaido": "北海道", "Aomori-ken": "青森県", "Iwate-ken": "岩手県",
    "Miyagi-ken": "宮城県", "Akita-ken": "秋田県", "Yamagata-ken": "山形県",
    "Fukushima-ken": "福島県", "Ibaraki-ken": "茨城県", "Tochigi-ken": "栃木県",
    "Gunma-ken": "群馬県", "Gumma-ken": "群馬県", "Saitama-ken": "埼玉県", "Chiba-ken": "千葉県",
    "Tokyo-to": "東京都", "Kanagawa-ken": "神奈川県", "Niigata-ken": "新潟県",
    "Toyama-ken": "富山県", "Ishikawa-ken": "石川県", "Fukui-ken": "福井県",
    "Yamanashi-ken": "山梨県", "Nagano-ken": "長野県", "Gifu-ken": "岐阜県",
    "Shizuoka-ken": "静岡県", "Aichi-ken": "愛知県", "Mie-ken": "三重県",
    "Shiga-ken": "滋賀県", "Kyoto-fu": "京都府", "Osaka-fu": "大阪府",
    "Hyogo-ken": "兵庫県", "Nara-ken": "奈良県", "Wakayama-ken": "和歌山県",
    "Tottori-ken": "鳥取県", "Shimane-ken": "島根県", "Okayama-ken": "岡山県",
    "Hiroshima-ken": "広島県", "Yamaguchi-ken": "山口県", "Tokushima-ken": "徳島県",
    "Kagawa-ken": "香川県", "Ehime-ken": "愛媛県", "Kochi-ken": "高知県",
    "Fukuoka-ken": "福岡県", "Saga-ken": "佐賀県", "Nagasaki-ken": "長崎県",
    "Kumamoto-ken": "熊本県", "Oita-ken": "大分県", "Miyazaki-ken": "宮崎県",
    "Kagoshima-ken": "鹿児島県", "Okinawa-ken": "沖縄県",
}

PREFECTURES = list(PREFECTURE_EN_TO_JP.values())

# 地方区分（可視化用）
REGIONS = {
    "北海道": "北海道",
    "青森県": "東北", "岩手県": "東北", "宮城県": "東北", "秋田県": "東北", "山形県": "東北", "福島県": "東北",
    "茨城県": "関東", "栃木県": "関東", "群馬県": "関東", "埼玉県": "関東", "千葉県": "関東", "東京都": "関東", "神奈川県": "関東",
    "新潟県": "中部", "富山県": "中部", "石川県": "中部", "福井県": "中部", "山梨県": "中部", "長野県": "中部", "岐阜県": "中部", "静岡県": "中部", "愛知県": "中部",
    "三重県": "近畿", "滋賀県": "近畿", "京都府": "近畿", "大阪府": "近畿", "兵庫県": "近畿", "奈良県": "近畿", "和歌山県": "近畿",
    "鳥取県": "中国", "島根県": "中国", "岡山県": "中国", "広島県": "中国", "山口県": "中国",
    "徳島県": "四国", "香川県": "四国", "愛媛県": "四国", "高知県": "四国",
    "福岡県": "九州", "佐賀県": "九州", "長崎県": "九州", "熊本県": "九州", "大分県": "九州", "宮崎県": "九州", "鹿児島県": "九州", "沖縄県": "九州",
}

REGION_COLORS = {
    "北海道": "#1f77b4",
    "東北": "#ff7f0e",
    "関東": "#2ca02c",
    "中部": "#d62728",
    "近畿": "#9467bd",
    "中国": "#8c564b",
    "四国": "#e377c2",
    "九州": "#7f7f7f",
}

REGION_MARKERS = {
    "北海道": "o",
    "東北": "s",
    "関東": "^",
    "中部": "D",
    "近畿": "v",
    "中国": "p",
    "四国": "h",
    "九州": "*",
}

# e-Stat「統計でみる都道府県のすがた2024」のダウンロード情報
ESTAT_FILES = {
    "A_population": "000040133641",
    # B_nature（自然環境）は気温・降水量等の地理的指標を含むため除外
    "C_economy": "000040133643",
    "D_administration": "000040133644",
    "E_education": "000040133645",
    "F_labor": "000040133646",
    "G_culture": "000040133647",
    "H_housing": "000040133648",
    "I_health": "000040133649",
    "J_welfare": "000040133650",
    "K_safety": "000040133651",
    "L_household": "000040133652",
}


def download_estat_data():
    """e-Statから統計データをダウンロード"""
    base_url = "https://www.e-stat.go.jp/stat-search/file-download?statInfId={}&fileKind=0"

    for name, stat_id in ESTAT_FILES.items():
        filepath = DATA_DIR / f"{name}.xls"
        if not filepath.exists():
            url = base_url.format(stat_id)
            print(f"ダウンロード中: {name}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                content = response.read()
                with open(filepath, "wb") as f:
                    f.write(content)
                print(f"  保存: {filepath} ({len(content)} bytes)")
        else:
            print(f"既存: {filepath}")


def parse_estat_excel(filepath):
    """
    e-Statの都道府県のすがたExcelファイルを解析し、
    47都道府県 x 指標のDataFrameを返す。

    ファイル構造:
    - Row 7: 英語ヘッダー（指標名）
    - Row 8: 指標コード（#A011000等）
    - 「Hokkaido」を含む行から47行が都道府県データ
    - 指標値は奇数列（11, 13, 15, ...）、順位は偶数列
    """
    xl = pd.ExcelFile(filepath)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(filepath, sheet_name=sheet, header=None)

    # 英語ヘッダー行（row 7）から指標名を取得
    header_row = 7
    indicator_cols = []
    indicator_names = []
    for j in range(11, len(df.columns), 2):
        name = df.iloc[header_row, j]
        if pd.notna(name):
            indicator_cols.append(j)
            indicator_names.append(str(name).strip().replace("\n", " "))

    # 「Hokkaido」を含む行を探す
    hokkaido_row = None
    for i in range(len(df)):
        for j in range(min(10, len(df.columns))):
            v = df.iloc[i, j]
            if pd.notna(v) and "Hokkaido" in str(v):
                hokkaido_row = i
                break
        if hokkaido_row is not None:
            break

    if hokkaido_row is None:
        print(f"  警告: Hokkaidoが見つかりません: {filepath}")
        return None

    # 英語都道府県名の列を特定（Hokkaidoがある列）
    en_name_col = None
    for j in range(10):
        v = df.iloc[hokkaido_row, j]
        if pd.notna(v) and "Hokkaido" in str(v):
            en_name_col = j
            break

    # 47都道府県分のデータを抽出
    data = {}
    for idx in range(47):
        row = hokkaido_row + idx
        if row >= len(df):
            break

        en_name = str(df.iloc[row, en_name_col]).strip()
        jp_name = PREFECTURE_EN_TO_JP.get(en_name)
        if jp_name is None:
            print(f"  警告: 都道府県名が対応しません: {en_name}")
            continue

        values = {}
        for col, name in zip(indicator_cols, indicator_names):
            v = df.iloc[row, col]
            if pd.notna(v):
                try:
                    values[name] = float(v)
                except (ValueError, TypeError):
                    pass
        data[jp_name] = values

    result = pd.DataFrame(data).T
    result.index.name = "都道府県"
    return result


def load_estat_data():
    """全カテゴリのe-Statデータを読み込み・統合"""
    download_estat_data()

    all_dfs = []
    for name in ESTAT_FILES.keys():
        filepath = DATA_DIR / f"{name}.xls"
        print(f"解析中: {name}...")
        df = parse_estat_excel(filepath)
        if df is not None:
            print(f"  {len(df)} 都道府県, {len(df.columns)} 指標")
            all_dfs.append(df)

    combined = pd.concat(all_dfs, axis=1)

    # 重複列名がある場合はサフィックスで区別
    if combined.columns.duplicated().any():
        cols = combined.columns.tolist()
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        combined.columns = new_cols

    print(f"\n統合結果: {combined.shape[0]} 都道府県, {combined.shape[1]} 指標")

    # 欠損が多すぎる列を除外（80%以上のデータがある列のみ残す）
    threshold = len(combined) * 0.8
    combined = combined.dropna(axis=1, thresh=int(threshold))
    print(f"欠損除外後: {combined.shape[1]} 指標")

    # 残った欠損値を列の中央値で補完
    combined = combined.fillna(combined.median())

    return combined


def run_umap(df, n_neighbors=10, min_dist=0.05, random_state=42):
    """UMAPで次元圧縮を実行"""
    from umap import UMAP
    from sklearn.preprocessing import StandardScaler

    print(f"使用する指標数: {len(df.columns)}")

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    # UMAP
    umap_model = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='euclidean'
    )
    X_umap = umap_model.fit_transform(X_scaled)

    return X_umap, df.columns.tolist()


def plot_umap_result(X_umap, prefectures, title="都道府県のUMAP可視化", filename="12_prefecture_umap.png"):
    """UMAP結果を可視化"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 地方ごとにプロット
    for region in REGION_COLORS.keys():
        mask = [REGIONS[p] == region for p in prefectures]
        indices = [i for i, m in enumerate(mask) if m]

        if indices:
            ax.scatter(
                X_umap[indices, 0],
                X_umap[indices, 1],
                c=REGION_COLORS[region],
                marker=REGION_MARKERS[region],
                s=100,
                label=region,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )

    # 都道府県名をラベル
    for i, pref in enumerate(prefectures):
        ax.annotate(
            pref,
            (X_umap[i, 0], X_umap[i, 1]),
            xytext=(5, -10),
            textcoords='offset points',
            fontsize=10,
            alpha=0.8
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_comparison(X_umap, prefectures):
    """地理的配置とUMAP配置の比較図を作成"""
    # 簡易的な緯度経度（概算）
    geo_coords = {
        "北海道": (43.0, 141.3), "青森県": (40.8, 140.7), "岩手県": (39.7, 141.1),
        "宮城県": (38.3, 140.9), "秋田県": (39.7, 140.1), "山形県": (38.2, 140.3),
        "福島県": (37.7, 140.5), "茨城県": (36.3, 140.4), "栃木県": (36.6, 139.9),
        "群馬県": (36.4, 139.1), "埼玉県": (35.9, 139.6), "千葉県": (35.6, 140.1),
        "東京都": (35.7, 139.7), "神奈川県": (35.4, 139.6), "新潟県": (37.9, 139.0),
        "富山県": (36.7, 137.2), "石川県": (36.6, 136.6), "福井県": (36.1, 136.2),
        "山梨県": (35.7, 138.6), "長野県": (36.7, 138.2), "岐阜県": (35.4, 136.8),
        "静岡県": (34.9, 138.4), "愛知県": (35.2, 137.0), "三重県": (34.7, 136.5),
        "滋賀県": (35.0, 136.0), "京都府": (35.0, 135.8), "大阪府": (34.7, 135.5),
        "兵庫県": (34.7, 135.2), "奈良県": (34.7, 135.8), "和歌山県": (34.2, 135.2),
        "鳥取県": (35.5, 134.2), "島根県": (35.5, 133.1), "岡山県": (34.7, 133.9),
        "広島県": (34.4, 132.5), "山口県": (34.2, 131.5), "徳島県": (34.1, 134.6),
        "香川県": (34.3, 134.0), "愛媛県": (33.8, 132.8), "高知県": (33.6, 133.5),
        "福岡県": (33.6, 130.4), "佐賀県": (33.2, 130.3), "長崎県": (32.7, 129.9),
        "熊本県": (32.8, 130.7), "大分県": (33.2, 131.6), "宮崎県": (31.9, 131.4),
        "鹿児島県": (31.6, 130.6), "沖縄県": (26.2, 127.7),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左: 地理的配置
    ax = axes[0]
    for region in REGION_COLORS.keys():
        mask = [REGIONS[p] == region for p in prefectures]
        indices = [i for i, m in enumerate(mask) if m]

        if indices:
            lons = [geo_coords[prefectures[i]][1] for i in indices]
            lats = [geo_coords[prefectures[i]][0] for i in indices]
            ax.scatter(
                lons, lats,
                c=REGION_COLORS[region],
                marker=REGION_MARKERS[region],
                s=100,
                label=region,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )

    ax.set_xlabel('経度', fontsize=12)
    ax.set_ylabel('緯度', fontsize=12)
    ax.set_title('地理的配置', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右: UMAP配置
    ax = axes[1]
    for region in REGION_COLORS.keys():
        mask = [REGIONS[p] == region for p in prefectures]
        indices = [i for i, m in enumerate(mask) if m]

        if indices:
            ax.scatter(
                X_umap[indices, 0],
                X_umap[indices, 1],
                c=REGION_COLORS[region],
                marker=REGION_MARKERS[region],
                s=100,
                label=region,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )

    # ラベル追加
    for i, pref in enumerate(prefectures):
        ax.annotate(
            pref,
            (X_umap[i, 0], X_umap[i, 1]),
            xytext=(5, -10),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('統計指標による配置（UMAP）', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('地理的位置と統計的類似性の比較', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_prefecture_umap_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: 12_prefecture_umap_comparison.png")


def main():
    """メイン処理"""
    print("=" * 60)
    print("都道府県データのUMAP可視化")
    print("データ出典: 総務省統計局「統計でみる都道府県のすがた2024」")
    print("=" * 60)

    # e-Statデータ読み込み
    df = load_estat_data()
    print(f"\nデータ形状: {df.shape}")
    print(f"指標数: {len(df.columns)}")

    # UMAP実行
    print("\nUMAP実行中...")
    X_umap, used_features = run_umap(df)
    print(f"使用した指標数: {len(used_features)}")

    # 可視化
    print("\n可視化...")
    prefectures = df.index.tolist()

    # UMAP単体
    plot_umap_result(X_umap, prefectures)

    # 地理との比較
    plot_comparison(X_umap, prefectures)

    print("\n完了!")


if __name__ == "__main__":
    main()
