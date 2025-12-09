"""
LLMによるオープンクエッション解析のデモ
自由記述の意見をLLMで分類・要約する例
"""

from openai import OpenAI

# 架空の市民意見（自由記述）
opinions = [
    "保育園の待機児童問題が深刻です。共働きなのに預け先がなく、妻が退職せざるを得ませんでした。",
    "高齢の母の介護と仕事の両立が難しい。デイサービスの枠を増やしてほしい。",
    "バスの本数が少なすぎて車がないと生活できない。免許返納した父が外出できなくなった。",
    "子どもの医療費無償化はありがたいが、小児科が少なすぎて予約が取れない。",
    "駅前の再開発より先に、老朽化した学校の建て替えを優先すべきだと思います。",
    "ゴミ収集の回数を減らすのは反対。夏場は衛生面で問題がある。",
    "公園の遊具が撤去されて子どもの遊び場がない。安全対策をした上で残してほしい。",
    "災害時の避難所が遠すぎる。高齢者は歩いて行けない。",
]

client = OpenAI()

# 1. 各意見の論点を抽出
print("=== 各意見の論点抽出 ===\n")
for i, opinion in enumerate(opinions, 1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"以下の市民意見から、主要な論点を1つ、10文字以内で抽出してください。\n\n意見: {opinion}"
        }]
    )
    topic = response.choices[0].message.content
    print(f"{i}. {topic}")
    print(f"   元の意見: {opinion[:30]}...\n")

# 2. 全体の傾向を要約
print("\n=== 全体の傾向分析 ===\n")
all_opinions = "\n".join([f"- {op}" for op in opinions])
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": f"""以下の市民意見を分析し、共通するテーマを3つ挙げてください。
各テーマについて、関連する意見の数も示してください。

{all_opinions}"""
    }]
)
print(response.choices[0].message.content)

# 3. 見落とされがちな少数意見の発見
print("\n=== 少数だが重要な指摘 ===\n")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": f"""以下の市民意見の中から、他とは異なる視点や、
見落とされがちだが重要な指摘を1つ選び、なぜ重要かを説明してください。

{all_opinions}"""
    }]
)
print(response.choices[0].message.content)
