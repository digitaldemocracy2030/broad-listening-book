"""
ミニ広聴AI - 意見のクラスタリングと可視化
"""
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# 初期化
load_dotenv("scripts/.env")
client = OpenAI()
plt.rcParams['font.family'] = 'MS Gothic'  # Windows用。Macは'Hiragino Sans'

# 1. データ準備
opinions = [
    "公園をもっと増やしてほしい",
    "緑地の保全に力を入れるべき",
    "子供が遊べる広場が足りない",
    "駅前の商業施設を充実させてほしい",
    "ショッピングモールが欲しい",
    "若者向けの店が少ない",
    "保育園の待機児童をなくしてほしい",
    "学童保育の時間を延長してほしい",
    "子育て支援を充実させるべき",
    "高齢者向けの施設が不足している",
    "バリアフリー化を進めてほしい",
    "福祉サービスの質を向上させるべき",
    "地元企業への就職支援を強化してほしい",
    "若者が働ける場所を増やすべき",
    "リモートワーク環境の整備を進めてほしい",
    "最低賃金を引き上げるべき",
]
df = pd.DataFrame({"opinion": opinions})

# 2. エンベディング取得
response = client.embeddings.create(
    input=df["opinion"].tolist(),
    model="text-embedding-3-small"
)
embeddings = np.array([item.embedding for item in response.data])

# 3. 次元圧縮 & クラスタリング
coords_2d = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(embeddings)
df["cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(coords_2d)
df["x"], df["y"] = coords_2d[:, 0], coords_2d[:, 1]

# 4. LLMでラベリング
cluster_labels = {}
for cluster_id in sorted(df["cluster"].unique()):
    opinions_in_cluster = df[df["cluster"] == cluster_id]["opinion"].tolist()
    opinions_text = "\n".join([f"- {op}" for op in opinions_in_cluster])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "あなたはKJ法が得意なデータ分析者です。与えられた意見グループに具体的で簡潔なラベル（10文字以内）を付けてください。ラベルのみを出力してください。"
        }, {
            "role": "user",
            "content": f"以下の意見グループにラベルを付けてください：\n\n{opinions_text}"
        }]
    )
    label = response.choices[0].message.content.strip()
    cluster_labels[cluster_id] = label
    print(f"クラスタ {cluster_id}: {label}")

# 5. 可視化
from scipy.spatial import ConvexHull

fig, ax = plt.subplots(figsize=(14, 10))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']  # 丸、四角、三角、ひし形
for cluster_id in range(4):
    mask = df["cluster"] == cluster_id
    points = df.loc[mask, ["x", "y"]].values
    ax.scatter(points[:, 0], points[:, 1],
               c=colors[cluster_id], marker=markers[cluster_id],
               label=cluster_labels[cluster_id], s=100, alpha=0.7)
    # 凸包で囲む
    if len(points) >= 3:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], c=colors[cluster_id], alpha=0.3)
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                c=colors[cluster_id], alpha=0.1)
    for _, row in df[mask].iterrows():
        ax.annotate(row["opinion"], (row["x"], row["y"]), fontsize=10)

ax.legend(fontsize=12)
ax.set_title("意見の可視化（ミニ広聴AI）", fontsize=14)
plt.tight_layout()
plt.savefig("images/14_mini_kouchou_ai.png", dpi=150)
print("図を保存しました: images/14_mini_kouchou_ai.png")
plt.show()
