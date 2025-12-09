"""
羅生門のワードクラウドデモ
青空文庫から羅生門のテキストを取得し、Janomeで形態素解析して
名詞と動詞のみでワードクラウドを作成する
"""

import urllib.request
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from janome.tokenizer import Tokenizer
import matplotlib

# 日本語フォントの設定（グローバル）
matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 青空文庫から羅生門のテキストを取得
url = "https://www.aozora.gr.jp/cards/000879/files/127_15260.html"

print("羅生門のテキストを取得中...")
with urllib.request.urlopen(url) as response:
    html = response.read().decode('shift_jis')

# HTMLからルビや注釈を除去して本文を抽出
# 青空文庫の形式に対応
text = re.sub(r'<ruby><rb>([^<]+)</rb><rp>[^<]*</rp><rt>[^<]*</rt><rp>[^<]*</rp></ruby>', r'\1', html)
text = re.sub(r'<[^>]+>', '', text)  # HTMLタグを除去
text = re.sub(r'［[^］]+］', '', text)  # 注釈を除去
text = re.sub(r'《[^》]+》', '', text)  # ルビを除去
text = re.sub(r'｜', '', text)  # ルビ開始記号を除去
text = re.sub(r'\s+', '', text)  # 空白を除去

# 本文部分を抽出（タイトルや著者名などを除く）
# 「ある日の暮方」から始まる
start_match = re.search(r'ある日の暮方', text)
if start_match:
    text = text[start_match.start():]

print(f"テキスト長: {len(text)}文字")
print(f"冒頭100文字: {text[:100]}...")

# Janomeで形態素解析
print("\n形態素解析中...")
tokenizer = Tokenizer()
tokens = tokenizer.tokenize(text)

# 名詞、動詞、形容詞を抽出
# 除外する単語（頻出するが意味のない語）
stopwords = {'する', 'いる', 'ある', 'なる', 'れる', 'られる', 'せる', 'させる'}

words = []
for token in tokens:
    pos = token.part_of_speech.split(',')[0]
    # 名詞、動詞、形容詞を抽出
    if pos in ['名詞', '動詞', '形容詞']:
        # 基本形を使用（動詞の活用形を統一）
        base_form = token.base_form if token.base_form != '*' else token.surface
        # 1文字の単語、記号、ストップワードは除外
        if len(base_form) > 1 and not re.match(r'^[ぁ-ん]$', base_form) and base_form not in stopwords:
            words.append(base_form)

print(f"抽出した単語数: {len(words)}")

# 単語の出現回数をカウント
word_counts = Counter(words)

# 出現回数順にソート
sorted_words = word_counts.most_common()

print("\n=== 上位20語 ===")
for word, count in sorted_words[:20]:
    print(f"  {word}: {count}回")

# 図を作成（左右に並べる）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- 左側：ワードクラウド ---
word_freq = dict(word_counts)

wordcloud = WordCloud(
    font_path='C:/Windows/Fonts/msgothic.ttc',
    width=700,
    height=500,
    background_color='white',
    colormap='copper',  # 羅生門の雰囲気に合わせた色
    max_words=100,
    min_font_size=10,
).generate_from_frequencies(word_freq)

ax1.imshow(wordcloud, interpolation='bilinear')
ax1.axis('off')
ax1.set_title('ワードクラウド：芥川龍之介「羅生門」', fontsize=14, fontweight='bold')

# --- 右側：単語と出現回数のリスト ---
ax2.axis('off')

# テーブルとして表示
table_data = [[word, str(count)] for word, count in sorted_words[:20]]
table = ax2.table(
    cellText=table_data,
    colLabels=['単語', '出現回数'],
    loc='center',
    cellLoc='center',
    colWidths=[0.4, 0.3]
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.6)

# 全セルにフォントを設定
for key, cell in table.get_celld().items():
    cell.set_text_props(fontfamily='MS Gothic')

# ヘッダーのスタイル
for i in range(2):
    table[(0, i)].set_facecolor('#8B4513')  # 茶色（羅生門の雰囲気）
    table[(0, i)].set_text_props(color='white', fontweight='bold', fontfamily='MS Gothic')

# 交互に色をつける
for i in range(1, len(table_data) + 1):
    if i % 2 == 0:
        for j in range(2):
            table[(i, j)].set_facecolor('#F5DEB3')  # 小麦色

plt.tight_layout()
plt.savefig('rashomon_wordcloud.png', dpi=150, bbox_inches='tight')
print("\n画像を保存しました: rashomon_wordcloud.png")
plt.show()

print("\n" + "="*50)
print("【ワードクラウドの限界】")
print("="*50)
print("""
このワードクラウドから「羅生門」の内容がわかりますか？

「下人」「老婆」「死骸」「髪」などの単語が並んでいますが、
これだけでは物語の本質は見えてきません。

- 下人はなぜ羅生門にいるのか？
- 老婆は何をしていたのか？
- 下人は最終的にどんな決断をしたのか？
- この物語のテーマは何か？

ワードクラウドは「どんな単語が多く使われているか」は示せても、
「その物語が何を語っているか」は教えてくれないのです。
""")
