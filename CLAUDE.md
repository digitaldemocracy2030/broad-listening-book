# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a book manuscript repository for "選挙を変えたブロードリスニング 生成AIが実現する民意の可視化と分析" (Broad Listening That Changed Elections: Visualization and Analysis of Public Opinion Using Generative AI), to be published by Impress.

The book covers "Broad Listening" technology - AI-powered systems for collecting, clustering, and analyzing large-scale public opinions, developed as part of Digital Democracy 2030 (DD2030).

## Repository Structure

- **Numbered markdown files (00-13)**: Book chapters in Japanese
  - 00_序文.md: Preface
  - 01-03: Part 1 - Concepts (What is Broad Listening)
  - 04-11: Part 2 - Case Studies
    - 04: 日本国内におけるブロードリスニングの広がり（安野氏の取り組み、国民民主、日テレ特番、Polis、朝日新聞）
    - 05: 東京都、シン東京2050
    - 06: 2025年参院選（チームみらい）
    - 07: DD2030による広聴AIの開発活動
    - 08: 地方選挙・地方自治体での活用
    - 09: 企業・NPOでの活用
    - 10: 海外におけるブロードリスニング
    - 11: 支援組織
  - 12-13: Part 3 - Technical explanations
- **column/**: Column articles for the book
- **images/**: Chapter images, named as `章番号_内容.png` (e.g., `01_broadlistening.png`)
- **code/**: Python demo scripts for opinion analysis
- **interview_questions/**: Interview question drafts
- **scripts/**: Build scripts
  - `build_pdf.py`: Markdown to HTML/PDF conversion script
- **book_order.txt**: Defines chapter order for PDF generation

## Writing Conventions

- **Format**: A5, ~270 pages, ~1000 characters per page
- **Language**: Japanese
- **Image format**: PNG preferred
- **Image naming**: `{chapter_number}_{description}.png`
- **License**: CC BY 4.0

## Style Guidelines

- **ダッシュの使用を避ける**: 文章を「——」（ダッシュ）で繋げることを避けてください。代わりに、句点で文を区切る、接続詞を使う、または文構造を工夫して表現してください。

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
