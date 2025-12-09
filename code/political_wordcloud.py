"""
政治的なワードクラウドデモ
架空のパブリックコメントをイメージした単語でワードクラウドを作成する
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import matplotlib

# 日本語フォントの設定（グローバル）
matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 架空のパブリックコメントから抽出したような単語
# 賛成派と反対派の意見が混在している想定
words_data = {
    # 頻出する中立的な単語
    '政策': 45,
    '市民': 42,
    '必要': 40,
    '問題': 38,
    '対策': 35,
    '地域': 32,
    '安全': 30,
    '環境': 28,
    '経済': 27,
    '生活': 25,
    '将来': 24,
    '子ども': 23,
    '健康': 22,
    '負担': 20,
    '影響': 19,
    '規制': 18,
    '改善': 17,
    '不安': 16,
    '推進': 15,
    '反対': 14,
    '賛成': 14,
    '住民': 13,
    '説明': 12,
    '検討': 12,
    '意見': 11,
    '計画': 11,
    '保護': 10,
    '開発': 10,
    '税金': 9,
    '予算': 9,
    '責任': 8,
    '透明性': 8,
    '参加': 7,
    '情報': 7,
    '理解': 6,
    '協力': 6,
    '議論': 5,
    '慎重': 5,
}

# 図を作成
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# ワードクラウドを生成
wordcloud = WordCloud(
    font_path='C:/Windows/Fonts/msgothic.ttc',
    width=800,
    height=600,
    background_color='white',
    colormap='RdYlBu',  # 赤・黄・青のカラーマップ（政治的な雰囲気）
    max_words=100,
    min_font_size=12,
    max_font_size=120,
).generate_from_frequencies(words_data)

ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')

plt.tight_layout()
plt.savefig('../images/political_wordcloud.png', dpi=150, bbox_inches='tight')
print("画像を保存しました: images/political_wordcloud.png")
plt.show()

print("\n=== このワードクラウドから何がわかる？ ===")
print("""
「政策」「市民」「必要」「問題」「対策」...

これらの単語が並んでいますが、
・この政策に賛成なのか反対なのか？
・何が問題だと言っているのか？
・どんな対策を求めているのか？

ワードクラウドからは何も読み取れません。
""")
