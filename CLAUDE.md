# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a book manuscript repository for "選挙を変えたブロードリスニング 生成AIが実現する民意の可視化と分析" (Broad Listening That Changed Elections: Visualization and Analysis of Public Opinion Using Generative AI), to be published by Impress.

The book covers "Broad Listening" technology - AI-powered systems for collecting, clustering, and analyzing large-scale public opinions, developed as part of Digital Democracy 2030 (DD2030).

## Repository Structure

- **Numbered markdown files (00-13)**: Book chapters in Japanese
  - 00_序文.md: Preface
  - 01-04: Part 1 - Concepts (What is Broad Listening)
  - 05-10: Part 2 - Case Studies (elections, government, corporate use)
  - 11-13: Part 3 - Technical explanations
- **column/**: Column articles for the book
- **images/**: Chapter images, named as `章番号_内容.png` (e.g., `01_broadlistening.png`)
- **code/**: Python demo scripts for opinion analysis
- **interview_questions/**: Interview question drafts

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

## Sample Code

Python scripts in `/code/` demonstrate:
- `llm_opinion_analysis.py`: LLM-based opinion classification using OpenAI API
- `political_wordcloud.py`, `rashomon_wordcloud.py`: Word cloud generation

Requires OpenAI API key for LLM scripts.
