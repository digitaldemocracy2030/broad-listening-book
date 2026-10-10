# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a book manuscript repository for "政治・自治体・企業に広がるブロードリスニング" (Broad Listening Spreading Across Politics, Local Governments, and Business), to be published by Impress.

The book covers "Broad Listening" technology - AI-powered systems for collecting, clustering, and analyzing large-scale public opinions, developed as part of Digital Democracy 2030 (DD2030).

## Repository Structure

- **Numbered markdown files (00-13)**: Book chapters in Japanese
  - 00_序文.md: Preface
  - 01-03: Part 1 - Concepts (What is Broad Listening)
  - 04-11: Part 2 - Case Studies
    - 04: 日本国内におけるブロードリスニングの広がり（安野氏の取り組み、国民民主、日テレ特番、Polis、朝日新聞）
    - 05: 東京都、シン東京2050
    - 06: 国政選挙でのブロードリスニングの利用（チームみらい、日本維新の会、国民民主党、公明党）
    - 07: 地方選挙での活用
    - 08: 地方自治体での活用
    - 09: 企業での活用
    - 10: ビジネスになったブロードリスニング（DD2030、ブーツ、多元現実、Democracy X、litela）
    - 11: 海外におけるブロードリスニング（台湾、Polis、ボーリンググリーン、イスラエル・パレスチナ、Connective Action）
  - 12-13: Part 3 - Technical explanations
- **column/**: Column articles for the book
- **images/**: Chapter images, named as `章番号_内容.png` (e.g., `01_broadlistening.png`)
- **code/**: Python demo scripts for opinion analysis
- **interview_questions/**: Interview question drafts
- **scripts/**: Build scripts
  - `build_pdf.py`: Markdown to HTML/PDF conversion script
- **book_order.txt**: Defines chapter order for PDF generation
- **en/**: English translations of chapters. 日本語原稿を修正しても自動では同期されない。日本語側を直したら英語側の同期が別途必要（同期PRの慣例あり: codex/jp-en-sync）
- **contribution_matrix.md**: 章・コラムごとの文責（担当著者）一覧。ただし以下はmatrixの記載と異なり実際は西尾担当: `09_02_サイボウズ.md`、`11_01_台湾.md`、`column/レビューで試す広聴AI.md`
- **wip-nishio/**, **tokoroten_取材_wip/**, **commented_pdf/**, **transcribe/**: 作業用ファイル。原稿のレビュー・修正・一括置換の対象に含めないこと

## 校正・修正のワークフロー

- 修正は**文責（担当著者）別にブランチ・PRを分ける**。tokoroten担当分は本人がdiff/PRで直接確認し、他著者分は「【○○さん確認依頼】」のPRを作成して著者にレビューを依頼する（例: PR #131〜#141）
- 事実関係の修正（日付・数値・固有名詞・法令の条番号など）は、必ずWeb等で一次情報の裏取りをしてから行い、PR本文にソースを記載する
- 修正の指摘と適用は分ける: まずレビュー報告（指摘一覧）を出し、確認後に修正を適用する

## Writing Conventions

- **Format**: A5, ~270 pages, ~1000 characters per page
- **Language**: Japanese
- **Image format**: PNG preferred
- **Image naming**: `{chapter_number}_{description}.png`
- **License**: CC BY-NC 4.0 (commits up to tag `license-cc-by-4.0-final` were CC BY 4.0)

## Style Guidelines

原稿の執筆・推敲・レビューでは **`docs/writing_rules.md`（本書の日本語文章ルール）に従う**。作業前に全文を読むこと。

- 機械検査: `uv run python scripts/style_lint.py`（`--summary` で件数一覧、`--dup` で章をまたいだ同一文、`--info` で参考レベルも表示）
- レビュー・推敲の手順: `.claude/skills/japanese-writing/SKILL.md`
- ルールの要点: 意味と事実を変えない／同じ型の高密度な反復を避ける（否定→言い直し構文は1章2回、「のです」は1節2回、修辞疑問は1章1回、格言着地は1章1〜2箇所まで）／身振り言葉・空虚な形容・増幅語・翻訳調を避ける／ダッシュは使わない

### 用語・表記の決定事項

- 「クエスチョン」表記（「クエッション」は使わない）
- 公明党のアンケートは「**We connect**」（公明新聞の公式表記。connectは小文字）
- 公明党の機関名は「**政務調査会**」（「政策調査会」ではない）
- シン東京2050のクラスタ名は「お台場エンターテ**イン**メント」（東京都公式ページの表記）
- TTTCの開発元は米AI Objectives Institute（「台湾発」は誤り）
- 広島県知事は2025年11月から横田美香氏（湯崎氏は2024年までの取材記録にのみ登場）
- パブリックコメント（意見公募手続）の根拠は行政手続法**第39条**（第38条は一般原則）

## Key Technologies Referenced

The book explains these technologies used in Broad Listening systems:
- **Talk to the City (TTTC)**: Opinion clustering and visualization tool
- **広聴AI (Kōchō AI)**: Japanese broad listening AI system
- **Sentence-BERT**: Text vectorization for semantic similarity
- **UMAP**: Dimensionality reduction for visualization
- **LLM**: Text understanding, summarization, and generation

## PDF Generation

校正用PDFを生成するには:

```bash
# 初回セットアップ（Chromiumブラウザのインストール）
uv run playwright install chromium

# PDF生成
uv run python scripts/build_pdf.py --pdf

# HTML生成（ブラウザで確認）
uv run python scripts/build_pdf.py
```

生成されたファイルは `output/` ディレクトリに `broad-listening-book-YYYYMMDD.pdf` の形式で出力されます。

## Sample Code

Python scripts in `/code/` demonstrate:
- `llm_opinion_analysis.py`: LLM-based opinion classification using OpenAI API
- `political_wordcloud.py`, `rashomon_wordcloud.py`: Word cloud generation

Requires OpenAI API key for LLM scripts.
