#!/usr/bin/env python3
"""
Markdown書籍をHTML/PDFに変換するスクリプト

使い方:
    uv run python scripts/build_pdf.py              # HTML生成してブラウザで開く
    uv run python scripts/build_pdf.py --pdf        # PDF生成
    uv run python scripts/build_pdf.py --order book_order.txt
"""

import argparse
import re
import webbrowser
from datetime import datetime
from pathlib import Path

import markdown


def load_order_file(order_file: Path) -> list[str]:
    """順序定義ファイルを読み込む（ファイル名のフラットリストを返す）"""
    files = []
    with open(order_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 空行、コメント行、セクションヘッダをスキップ
            if not line or line.startswith("#") or re.match(r"^\[.+\]$", line):
                continue
            files.append(line)
    return files


def load_chapter_groups(order_file: Path) -> dict[str, list[str]]:
    """順序定義ファイルから章グループを読み込む

    [セクション名] で章の区切りとPDFファイル名を定義する。
    セクション名がそのまま出力PDFのファイル名になる（.pdfは自動付与）。
    """
    chapters: dict[str, list[str]] = {}
    chapter_order: list[str] = []
    current_section: str | None = None

    with open(order_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # セクションヘッダ: [section_name]
            section_match = re.match(r"^\[(.+)\]$", line)
            if section_match:
                current_section = section_match.group(1)
                if current_section not in chapters:
                    chapters[current_section] = []
                    chapter_order.append(current_section)
                continue
            # ファイル行
            if current_section is not None:
                chapters[current_section].append(line)

    return {k: chapters[k] for k in chapter_order}


def read_markdown_file(filepath: Path) -> str:
    """単一のMarkdownファイルを読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def convert_image_paths(html: str, source_file: Path, base_dir: Path) -> str:
    """画像パスを絶対パスに変換

    優先順位:
    1) Markdownファイルからの相対パス
    2) リポジトリルートからの相対パス（既存原稿との互換用）
    """

    def resolve_local_path(src: str) -> Path:
        local_candidate = (source_file.parent / src).resolve()
        if local_candidate.exists():
            return local_candidate

        fallback_candidate = (base_dir / src).resolve()
        if fallback_candidate.exists():
            return fallback_candidate

        # どちらにも存在しない場合は、従来に近い挙動として
        # Markdownファイル基準のパスをそのまま使う。
        return local_candidate

    def replace_src(match):
        src = match.group(1)
        if src.startswith(("http://", "https://", "file://", "data:", "/")):
            return match.group(0)

        if not src.startswith(("http://", "https://", "file://")):
            abs_path = resolve_local_path(src)
            # Windowsパスをfile:// URLに変換
            file_url = abs_path.as_uri()
            return f'src="{file_url}"'
        return match.group(0)

    html = re.sub(r'src="([^"]+)"', replace_src, html)
    return html


def markdown_to_html(md_content: str) -> str:
    """MarkdownをHTMLに変換"""
    extensions = [
        "tables",
        "fenced_code",
        "footnotes",
        "toc",
    ]
    return markdown.markdown(md_content, extensions=extensions)


def get_css(for_pdf: bool = False) -> str:
    """PDF/HTML用のCSSスタイル"""
    preview_notice = "" if for_pdf else """
    .no-print {
        background: #ffffcc;
        padding: 10px;
        margin-bottom: 20px;
        border: 1px solid #ccc;
    }
    """

    return f"""
    @page {{
        size: A5;
        margin: 20mm 15mm 25mm 15mm;
    }}

    @media print {{
        body {{
            font-size: 7pt;
        }}
        hr.chapter-break {{
            page-break-after: always;
            visibility: hidden;
        }}
        h1 {{
            page-break-before: always;
        }}
        h1:first-of-type {{
            page-break-before: avoid;
        }}
        .no-print {{
            display: none;
        }}
    }}

    body {{
        font-family: "Hiragino Kaku Gothic ProN", "Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", sans-serif;
        font-size: 7pt;
        line-height: 1.8;
        text-align: justify;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }}

    h1 {{
        font-size: 12pt;
        margin-top: 40px;
        margin-bottom: 20px;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }}

    h2 {{
        font-size: 9pt;
        margin-top: 30px;
        margin-bottom: 15px;
        border-left: 4px solid #666;
        padding-left: 10px;
    }}

    h3 {{
        font-size: 8pt;
        margin-top: 20px;
        margin-bottom: 10px;
    }}

    p {{
        margin: 0 0 0.5em 0;
        text-indent: 1em;
    }}

    p:first-child,
    h1 + p, h2 + p, h3 + p,
    ul + p, ol + p, blockquote + p {{
        text-indent: 0;
    }}

    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 20px auto;
    }}

    blockquote {{
        margin: 20px 0;
        padding: 10px 20px;
        border-left: 3px solid #999;
        background-color: #f5f5f5;
        font-size: 7pt;
    }}

    code {{
        font-family: "Consolas", "Monaco", "Courier New", monospace;
        font-size: 6pt;
        background-color: #f0f0f0;
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }}

    pre {{
        background-color: #f5f5f5;
        padding: 15px;
        overflow-x: auto;
        font-size: 6pt;
        line-height: 1.4;
        border-radius: 5px;
    }}

    pre code {{
        background-color: transparent;
        padding: 0;
    }}

    hr {{
        border: none;
        border-top: 1px solid #ccc;
        margin: 40px 0;
    }}

    hr.chapter-break {{
        border: none;
        margin: 0;
        padding: 0;
    }}

    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        font-size: 7pt;
    }}

    th, td {{
        border: 1px solid #ccc;
        padding: 8px 12px;
        text-align: left;
        overflow-wrap: anywhere;
    }}

    th {{
        background-color: #f0f0f0;
    }}

    ul, ol {{
        margin: 0.5em 0;
        padding-left: 2em;
    }}

    li {{
        margin: 0.2em 0;
    }}

    li > p {{
        margin: 0;
        text-indent: 0;
    }}

    /* 脚注 */
    .footnote {{
        font-size: 6pt;
        border-top: 1px solid #ccc;
        margin-top: 10px;
        padding-top: 5px;
    }}

    .footnote hr {{
        display: none;
    }}

    .footnote ol {{
        margin: 0;
        padding-left: 1.5em;
    }}

    .footnote li p {{
        margin: 0.2em 0;
    }}

    .footnote-ref {{
        font-size: 5pt;
        vertical-align: super;
    }}

    sup {{
        font-size: 5pt;
    }}

    /* 図版キャプション風（画像の次の行） */
    img + em,
    p > img:only-child + em {{
        display: block;
        text-align: center;
        font-size: 6pt;
        color: #666;
        margin-top: -10px;
    }}

    {preview_notice}
    """


def build_html_content_from_files(
    base_dir: Path, file_list: list[str], for_pdf: bool = False
) -> str:
    """ファイルリストからHTMLコンテンツを生成"""
    # 各ファイルを個別にHTMLに変換してから結合（脚注を各章内に保持するため）
    html_parts = []
    for filename in file_list:
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"警告: ファイルが見つかりません: {filepath}")
            continue

        # 個別のMarkdownをHTMLに変換
        md_content = read_markdown_file(filepath)
        html_part = markdown_to_html(md_content)

        # 画像パスを絶対パスに変換
        html_part = convert_image_paths(html_part, filepath, base_dir)

        html_parts.append(html_part)
        html_parts.append('<hr class="chapter-break">')  # 章の区切り

    html_body = "\n".join(html_parts)

    # プレビュー用のバナー（PDF生成時は含めない）
    preview_banner = "" if for_pdf else """
<div class="no-print">
    <strong>校正用プレビュー</strong> - ブラウザの印刷機能 (Ctrl+P) でPDFに変換できます。
    印刷設定で「A5」「余白なし」を選択してください。
</div>
"""

    # 完全なHTMLドキュメントを作成
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>選挙を変えたブロードリスニング</title>
    <style>
{get_css(for_pdf=for_pdf)}
    </style>
</head>
<body>
{preview_banner}
{html_body}
</body>
</html>
"""
    return html_content


def build_html_content(base_dir: Path, order_file: Path, for_pdf: bool = False) -> str:
    """HTMLコンテンツを生成"""
    file_list = load_order_file(order_file)
    print(f"対象ファイル数: {len(file_list)}")
    return build_html_content_from_files(base_dir, file_list, for_pdf)


def build_html(
    base_dir: Path,
    order_file: Path,
    output_file: Path,
    open_browser: bool = True,
) -> None:
    """HTMLを生成"""
    print(f"順序ファイル: {order_file}")
    print(f"出力先: {output_file}")

    html_content = build_html_content(base_dir, order_file, for_pdf=False)

    # 出力ディレクトリを作成
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # HTMLを保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"完了: {output_file}")

    # ブラウザで開く
    if open_browser:
        print("ブラウザで開いています...")
        webbrowser.open(output_file.as_uri())


def build_pdf(
    base_dir: Path,
    order_file: Path,
    output_file: Path,
) -> None:
    """PDFを生成（Playwright使用）"""
    from playwright.sync_api import sync_playwright

    print(f"順序ファイル: {order_file}")
    print(f"出力先: {output_file}")

    html_content = build_html_content(base_dir, order_file, for_pdf=True)

    # 出力ディレクトリを作成
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 一時HTMLファイルを作成
    temp_html = output_file.with_suffix(".html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("PDF生成中（Chromiumを使用）...")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # HTMLファイルを開く
        page.goto(temp_html.as_uri(), wait_until="networkidle")

        # PDFを生成（A5サイズ）
        page.pdf(
            path=str(output_file),
            format="A5",
            margin={
                "top": "20mm",
                "right": "15mm",
                "bottom": "25mm",
                "left": "15mm",
            },
            print_background=True,
        )

        browser.close()

    # 一時HTMLファイルを削除（オプション：残しておきたい場合はコメントアウト）
    # temp_html.unlink()

    print(f"完了: {output_file}")


def build_chapter_pdfs(
    base_dir: Path,
    order_file: Path,
    output_dir: Path,
) -> list[Path]:
    """章ごとのPDFを生成（Playwright使用）"""
    from playwright.sync_api import sync_playwright

    chapters = load_chapter_groups(order_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"章数: {len(chapters)}")
    pdf_files: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()

        for section_name, files in chapters.items():
            output_file = output_dir / f"{section_name}.pdf"
            print(f"章PDF生成中: {section_name}.pdf（{len(files)}ファイル）")

            html_content = build_html_content_from_files(base_dir, files, for_pdf=True)

            temp_html = output_file.with_suffix(".html")
            with open(temp_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            page = browser.new_page()
            page.goto(temp_html.as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(output_file),
                format="A5",
                margin={
                    "top": "20mm",
                    "right": "15mm",
                    "bottom": "25mm",
                    "left": "15mm",
                },
                print_background=True,
            )
            page.close()
            temp_html.unlink()

            pdf_files.append(output_file)

        browser.close()

    print(f"章PDF生成完了: {len(pdf_files)}ファイル → {output_dir}")
    return pdf_files


def print_char_count_table(base_dir: Path, order_file: Path) -> None:
    """各ファイルのUTF-8文字数をコンソール表示し、Markdown形式でファイル出力"""
    file_list = load_order_file(order_file)

    col_file = 55
    col_chars = 10
    width = col_file + col_chars
    sep = "-" * width

    # コンソール出力
    print(f"\n{'=' * width}")
    print(f"{'ファイル':<{col_file}}{'文字数':>{col_chars}}")
    print(sep)

    counts: list[tuple[str, int]] = []
    grand_total = 0
    for filename in file_list:
        filepath = base_dir / filename
        if filepath.exists():
            text = filepath.read_text(encoding="utf-8")
            char_count = len(text)
        else:
            char_count = 0
        grand_total += char_count
        counts.append((filename, char_count))
        print(f"{filename:<{col_file}}{char_count:>{col_chars},}")

    print(sep)
    print(f"{'合計':<{col_file}}{grand_total:>{col_chars},}")
    print(f"{'=' * width}")

    # Markdown形式でファイル出力
    md_lines = [
        "| ファイル | 文字数 |",
        "|:---|---:|",
    ]
    for filename, char_count in counts:
        md_lines.append(f"| {filename} | {char_count:,} |")
    md_lines.append(f"| **合計** | **{grand_total:,}** |")

    md_path = base_dir / "output" / "char_counts.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n文字数表: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Markdown書籍をHTML/PDFに変換")
    parser.add_argument(
        "--order",
        type=Path,
        default=Path("book_order.txt"),
        help="順序定義ファイル (デフォルト: book_order.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="出力ファイル (デフォルト: output/broad-listening-book-YYYYMMDD.html/pdf)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="PDFを生成（デフォルトはHTML）",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="HTML生成後にブラウザで開かない",
    )
    parser.add_argument(
        "--per-chapter",
        action="store_true",
        help="章ごとのPDFを生成（output/chapters/ に出力）",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    order_file = base_dir / args.order

    # ビルド日付を含むデフォルトファイル名を生成
    build_date = datetime.now().strftime("%Y%m%d")
    default_pdf = f"output/broad-listening-book-{build_date}.pdf"
    default_html = f"output/broad-listening-book-{build_date}.html"

    if args.per_chapter:
        chapter_output_dir = base_dir / "output" / "chapters"
        build_chapter_pdfs(base_dir, order_file, chapter_output_dir)

    if args.pdf:
        output_file = base_dir / (args.output or Path(default_pdf))
        build_pdf(base_dir, order_file, output_file)
    elif not args.per_chapter:
        output_file = base_dir / (args.output or Path(default_html))
        build_html(base_dir, order_file, output_file, open_browser=not args.no_open)

    # ビルド後に文字数一覧を表示
    print_char_count_table(base_dir, order_file)


if __name__ == "__main__":
    main()
