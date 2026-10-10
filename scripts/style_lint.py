#!/usr/bin/env python3
"""
本書の日本語文章ルール（docs/writing_rules.md）のうち、機械で数えられる項目を検出する。

指摘は推敲の手がかりで、合否判定ではない。格言着地・両論併記・論証の飛躍などは
人が読んで判断する。引用（「」内と > 引用ブロック）、コード、HTMLコメント、
画像行は検査対象から外す。

使い方:
    uv run python scripts/style_lint.py                # book_order.txt の全ファイル
    uv run python scripts/style_lint.py 02_*.md        # 指定ファイル
    uv run python scripts/style_lint.py --summary      # 件数一覧のみ
    uv run python scripts/style_lint.py --dup          # 全原稿をまたいだ同一文の検出
    uv run python scripts/style_lint.py --info         # 参考レベル（info）の指摘も表示
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- 行単位で検出するパターン（docs/writing_rules.md の節番号と対応） ---
# (rule_id, severity, regex, 説明)
LINE_RULES = [
    ("dash", "warn", r"——|―|──|━|—", "ダッシュ（§8）。括弧・句点・接続詞に直す"),
    ("gesture", "warn",
     r"注目すべき|重要なのは|大切なのは|特筆すべき|見逃せない|ここで注目したい|強調したいのは|ポイントは",
     "身振り言葉（§6.1）。指す内容を主語にする"),
    ("preface", "warn",
     r"見ていきましょう|見ていきます|見ていこう|掘り下げ|深掘り|紹介していきます|解説していきます",
     "予告だけの文（§6.1）"),
    ("empty-adj", "warn",
     r"正面から|不可欠|核心的|鍵となる|カギとなる|多角的|包括的|総合的に|真摯に向き合",
     "空虚な形容・姿勢の宣言（§6.2）"),
    ("kimebun", "warn",
     r"に他なら|にほかなら|と言っても過言|と言えるでしょう|と言えるだろう|これこそが|いかがでしたか|参考になれば幸い",
     "決め文・定型の結び（§6.2）"),
    ("amplify", "warn",
     r"劇的|飛躍的|計り知れ|画期的|驚くほど|目覚まし|極めて重要|非常に重要",
     "増幅語（§6.3）。数字が併記されていれば削る"),
    ("amplify-masani", "info", r"まさに|まさしく", "増幅語（§6.3）"),
    ("metaphor", "info",
     r"解像度|腹落ち|言語化|手触り|肌感|温度感|熱量|勝ち筋|刺さる|落とし込|紐解|土台|両輪|カギ|落とし穴",
     "比喩でぼかす語（§6.4）。指す内容を書けるなら具体的に"),
    ("katai", "info",
     r"ことができ|ことが可能|ことによって|という観点から|の観点から|という点において",
     "翻訳調・硬い言い回し（§6.5, §7）"),
    ("count-decl", "warn",
     r"(理由|ポイント|特徴|観点|要素|課題|論点|視点)は(主に|大きく)?[0-9０-９一二三四五六七八九]つ",
     "数の宣言（§2）。数を先に決めず内容から書く"),
    ("double-neg", "info", r"ないわけではな|なくはな|ないことはな|なくもな",
     "二重否定（§4）"),
    ("i-adj-desu", "info", r"(?<![たな])いです[。]", "イ形容詞＋です（§4）。前後の流れごと見直す"),
    ("no-chain", "info",
     r"[一-龯ァ-ヶーA-Za-z0-9]+の[一-龯ァ-ヶーA-Za-z0-9]+の[一-龯ァ-ヶーA-Za-z0-9]+の[一-龯ァ-ヶーA-Za-z0-9]+",
     "「の」の3連鎖（§4）"),
]
LINE_RULES = [(rid, sev, re.compile(rx), msg) for rid, sev, rx, msg in LINE_RULES]

# --- 件数に上限があるパターン（§5） ---
RE_ANOTB = re.compile(r"ではなく|だけでなく|だけではなく")
RE_NODESU = re.compile(r"のです。|のだ。|のである。")
RE_RHETORIC = re.compile(r"(では|それでは|なぜ|一体)[、]?[^。]*?(でしょうか|だろうか)[？?]")
RE_BOLD = re.compile(r"\*\*[^*\n]+?\*\*")
RE_BOLD_LABEL_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*\*\*[^*]+\*\*\s*[:：]?")
RE_DESUMASU = re.compile(r"(です|ます|ました|でした|ません|でしょう)。")
RE_DEARU = re.compile(r"(である|だ|ではない|[^せ]ない|[^し]た)。")

LIMIT_ANOTB_PER_FILE = 2
LIMIT_NODESU_PER_SECTION = 2
LIMIT_RHETORIC_PER_FILE = 1
LIMIT_BOLD_PER_SECTION = 2
LIMIT_BOLD_LABEL_LISTS_PER_FILE = 1
LONG_SENTENCE = 100
DUP_MIN_LEN = 25

RE_QUOTE = re.compile(r"「[^「」]*」|『[^『』]*』")
RE_INLINE_CODE = re.compile(r"`[^`]*`")
RE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
RE_URL = re.compile(r"https?://\S+")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*)")
RE_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")
# インタビューの発言行（**本間：** ...）。発言は引用として扱い検査しない
RE_SPEAKER = re.compile(
    r"^\s*(?:[-*+]\s*)?\*\*[^*\s（）()]{1,10}(?:[：:]\s*\*\*|\*\*\s*[：:]?\s)"
)
# 箇条書き先頭のラベル（- **用語**）と、行頭の定義ラベル（**用語**：）
RE_LIST_LABEL = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)])\s*\*\*[^*]+\*\*|\*\*[^*]+\*\*\s*[：:])")


@dataclass
class Finding:
    line: int
    rule: str
    severity: str
    message: str
    snippet: str


@dataclass
class Line:
    no: int
    raw: str
    prose: str  # 引用・記法を除いた地の文
    section: int
    heading: bool
    is_list: bool
    is_table: bool


@dataclass
class FileReport:
    path: Path
    findings: list[Finding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def load_order(order_file: Path) -> list[Path]:
    files = []
    for line in order_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        files.append(ROOT / line)
    return files


def strip_quotes(text: str) -> str:
    # 入れ子の「」に備えて内側から繰り返し外す
    prev = None
    while prev != text:
        prev = text
        text = RE_QUOTE.sub("「」", text)
    return text


def count_speaker_lines(text: str) -> int:
    return sum(1 for raw in text.splitlines() if RE_SPEAKER.match(raw.strip()))


def parse_lines(text: str) -> list[Line]:
    lines = []
    in_code = False
    in_comment = False
    section = 0
    for no, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if in_comment:
            if "-->" in s:
                in_comment = False
            continue
        if s.startswith("<!--"):
            if "-->" not in s:
                in_comment = True
            continue
        if not s or s.startswith(">") or s.startswith("![") or RE_SPEAKER.match(s):
            continue
        m = RE_HEADING.match(s)
        if m:
            section += 1
            lines.append(Line(no, raw, m.group(2), section, True, False, False))
            continue
        prose = RE_INLINE_CODE.sub("", s)
        prose = RE_LINK.sub(r"\1", prose)
        prose = RE_URL.sub("", prose)
        prose = RE_HTML_TAG.sub("", prose)
        prose = strip_quotes(prose)
        is_list = bool(re.match(r"^(?:[-*+]|\d+[.)])\s", s))
        is_table = s.startswith("|")
        lines.append(Line(no, raw, prose, section, False, is_list, is_table))
    return lines


def snippet(text: str, m: re.Match | None = None, width: int = 24) -> str:
    text = text.strip()
    if m is None:
        return text[: width * 2] + ("…" if len(text) > width * 2 else "")
    a = max(0, m.start() - width)
    b = min(len(text), m.end() + width)
    return ("…" if a > 0 else "") + text[a:b] + ("…" if b < len(text) else "")


def sentences(prose: str) -> list[str]:
    return [s for s in RE_SENT_SPLIT.split(prose) if s.strip()]


def lint_file(path: Path) -> FileReport:
    rep = FileReport(path)
    text = path.read_text(encoding="utf-8")
    lines = parse_lines(text)
    # 話者ラベルが3つ以上あればインタビュー記事とみなす（地の文と発言で文体が違ってよい）
    is_interview = count_speaker_lines(text) >= 3
    body = [ln for ln in lines if not ln.heading]

    def add(ln: Line, rule: str, sev: str, msg: str, snip: str):
        rep.findings.append(Finding(ln.no, rule, sev, msg, snip))
        rep.counts[rule] += 1

    # 行単位の語彙パターン
    for ln in body:
        if ln.is_table:
            continue
        for rid, sev, rx, msg in LINE_RULES:
            # ダッシュだけは「」内も含めて検出する（表記の問題なので）
            target = ln.raw if rid == "dash" else ln.prose
            for m in rx.finditer(target):
                add(ln, rid, sev, msg, snippet(target, m))

        if not ln.is_table:
            for s in sentences(ln.prose):
                n = len(re.sub(r"[\s*_#]", "", s))
                if n > LONG_SENTENCE:
                    add(ln, "long-sentence", "info",
                        f"{n}字の文（§4）。分けられないか見直す", snippet(s))

    # 否定してから言い直す構文: 1ファイルあたり
    hits = [(ln, m) for ln in body for m in RE_ANOTB.finditer(ln.prose)]
    rep.counts["a-not-b"] = len(hits)
    if len(hits) > LIMIT_ANOTB_PER_FILE:
        for ln, m in hits:
            add(ln, "a-not-b", "warn",
                f"否定→言い直し構文 {len(hits)}回/章（上限{LIMIT_ANOTB_PER_FILE}、§5）",
                snippet(ln.prose, m))
        rep.counts["a-not-b"] = len(hits)

    # 修辞疑問: 1ファイルあたり
    hits = [(ln, m) for ln in body for m in RE_RHETORIC.finditer(ln.prose)]
    rep.counts["rhetoric"] = len(hits)
    if len(hits) > LIMIT_RHETORIC_PER_FILE:
        for ln, m in hits:
            add(ln, "rhetoric", "warn",
                f"修辞疑問 {len(hits)}回/章（上限{LIMIT_RHETORIC_PER_FILE}、§5）",
                snippet(ln.prose, m))
        rep.counts["rhetoric"] = len(hits)

    # 節ごとの上限: のです／太字
    by_section: dict[int, list[Line]] = defaultdict(list)
    for ln in body:
        by_section[ln.section].append(ln)
    for sec_lines in by_section.values():
        hits = [(ln, m) for ln in sec_lines for m in RE_NODESU.finditer(ln.prose)]
        if len(hits) > LIMIT_NODESU_PER_SECTION:
            for ln, m in hits:
                add(ln, "nodesu", "warn",
                    f"「のです／のだ」{len(hits)}回/節（上限{LIMIT_NODESU_PER_SECTION}、§5）",
                    snippet(ln.prose, m))
        # 表と、箇条書き先頭のラベル（用語集など）は数えない
        hits = [(ln, m) for ln in sec_lines if not ln.is_table
                for m in RE_BOLD.finditer(RE_LIST_LABEL.sub("", ln.raw))]
        if len(hits) > LIMIT_BOLD_PER_SECTION:
            for ln, m in hits:
                add(ln, "bold", "warn",
                    f"太字 {len(hits)}箇所/節（上限{LIMIT_BOLD_PER_SECTION}、§5）",
                    snippet(ln.raw, m))

    # 太字ラベル付き箇条書きのまとまり（3項目以上）
    groups: list[list[Line]] = []
    cur: list[Line] = []
    prev_no = -10
    for ln in body:
        if RE_BOLD_LABEL_ITEM.match(ln.raw) and ln.no - prev_no <= 2:
            cur.append(ln)
        elif RE_BOLD_LABEL_ITEM.match(ln.raw):
            if len(cur) >= 3:
                groups.append(cur)
            cur = [ln]
        prev_no = ln.no if RE_BOLD_LABEL_ITEM.match(ln.raw) else prev_no
    if len(cur) >= 3:
        groups.append(cur)
    rep.counts["bold-label-list"] = len(groups)
    if len(groups) > LIMIT_BOLD_LABEL_LISTS_PER_FILE:
        for g in groups:
            add(g[0], "bold-label-list", "warn",
                f"太字ラベル付き箇条書き {len(groups)}組/章（上限{LIMIT_BOLD_LABEL_LISTS_PER_FILE}、§5）",
                snippet(g[0].raw))

    # 同じ書き出しの3連続（同一行・同一段落内）
    for ln in body:
        if ln.is_table or ln.is_list:
            continue
        heads = [s.strip()[:2] for s in sentences(ln.prose) if len(s.strip()) >= 4]
        for i in range(len(heads) - 2):
            a, b, c = heads[i:i + 3]
            same_digit = all(re.match(r"[0-9０-９]", h) for h in (a, b, c))
            if (a == b == c and not a.startswith("「")) or same_digit:
                add(ln, "same-opening", "info",
                    f"書き出し「{a}」が3文連続（§5）", snippet(ln.prose))
                break

    # 文体の混在
    ds = [(ln, m) for ln in body if not ln.is_table for m in RE_DESUMASU.finditer(ln.prose)]
    da = [(ln, m) for ln in body if not ln.is_table for m in RE_DEARU.finditer(ln.prose)]
    rep.counts["desumasu"] = len(ds)
    rep.counts["dearu"] = len(da)
    if ds and da and not is_interview:
        minor, major_name = (da, "です・ます") if len(ds) >= len(da) else (ds, "である")
        if len(minor) >= 3 and len(minor) / (len(ds) + len(da)) >= 0.05:
            for ln, m in minor[:10]:
                add(ln, "style-mix", "warn",
                    f"文体の混在（主: {major_name}、§1）。地の文なら統一する",
                    snippet(ln.prose, m))

    rep.findings.sort(key=lambda f: (f.line, f.rule))
    return rep


def find_duplicates(files: list[Path]) -> dict[str, list[tuple[Path, int]]]:
    occ: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for path in files:
        for ln in parse_lines(path.read_text(encoding="utf-8")):
            if ln.heading or ln.is_table:
                continue
            for s in sentences(ln.prose):
                key = re.sub(r"[\s、。，．・*「」『』（）()]", "", s)
                if len(key) >= DUP_MIN_LEN:
                    occ[key].append((path, ln.no))
    return {k: v for k, v in occ.items() if len(v) >= 2}


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description="本書の文章ルールの機械検査")
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--order", type=Path, default=ROOT / "book_order.txt")
    ap.add_argument("--summary", action="store_true", help="件数一覧のみ表示")
    ap.add_argument("--dup", action="store_true", help="全原稿をまたいだ同一文を検出")
    ap.add_argument("--info", action="store_true", help="info レベルの指摘も表示")
    args = ap.parse_args()

    files = args.files or load_order(args.order)
    files = [f for f in files if f.exists()]

    if args.dup:
        dups = find_duplicates(files)
        print(f"# 同一文（{DUP_MIN_LEN}字以上、{len(dups)}件）\n")
        for key, locs in sorted(dups.items(), key=lambda kv: -len(kv[1])):
            print(f"- {key[:60]}{'…' if len(key) > 60 else ''}")
            for p, no in locs:
                print(f"    {rel(p)}:{no}")
        return

    reports = [lint_file(f) for f in files]

    if args.summary:
        cols = ["a-not-b", "nodesu", "rhetoric", "bold", "gesture", "dash",
                "empty-adj", "kimebun", "amplify", "style-mix", "long-sentence"]
        print("| ファイル | 否定→言い直し | のです超過 | 修辞疑問 | 太字超過 | 身振り | ダッシュ | 空虚形容 | 決め文 | 増幅 | 文体混在 | 長文 |")
        print("|---|" + "---:|" * len(cols))
        for r in reports:
            print(f"| {rel(r.path)} | " + " | ".join(str(r.counts.get(c, 0)) for c in cols) + " |")
        return

    total_warn = 0
    for r in reports:
        shown = [f for f in r.findings if args.info or f.severity == "warn"]
        if not shown:
            continue
        total_warn += sum(1 for f in shown if f.severity == "warn")
        print(f"\n## {rel(r.path)}")
        for f in shown:
            mark = "W" if f.severity == "warn" else "i"
            print(f"  {rel(r.path)}:{f.line} [{mark}:{f.rule}] {f.message}\n      {f.snippet}")
    print(f"\n合計 warn {total_warn} 件（{len(reports)} ファイル）。"
          "lint の指摘ゼロはAIくさくない保証ではない。docs/writing_rules.md を参照。")


if __name__ == "__main__":
    main()
