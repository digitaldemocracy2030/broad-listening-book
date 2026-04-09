#!/usr/bin/env python3
"""
English translation tooling for the manuscript.

Examples:
    python3 scripts/english.py list-sections
    python3 scripts/english.py translate-section --first --dry-run
    python3 scripts/english.py translate-section --section ch00_preface --model gpt-5.4
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH_MAP = {
    "00_序文.md": "00_preface.md",
    "00_endorsement_from_Audrey.md": "00_endorsement_from_audrey_tang.md",
    "01_ブロードリスニングとは何か？.md": "01_what_is_broad_listening.md",
    "02_ブロードリスニングとアンケートの違い、定量分析から定性分析へ.md": "02_broad_listening_vs_surveys.md",
    "03_デジタル民主主義とブロードリスニング、新しい民意の届け方.md": "03_digital_democracy_and_new_public_voice.md",
    "04_日本国内におけるブロードリスニングの広がり.md": "04_spread_of_broad_listening_in_japan.md",
    "04_01_安野貴博の取り組み.md": "04_01_takahiro_anno_initiatives.md",
    "04_02_国民民主の国会質問.md": "04_02_dpfp_parliamentary_questions.md",
    "04_03_sumino_日テレ選挙特番.md": "04_03_sumino_ntv_election_special.md",
    "04_04_Polisで世論の地図を作る.md": "04_04_mapping_public_opinion_with_polis.md",
    "04_05_朝日新聞の特設記事.md": "04_05_asahi_special_feature.md",
    "05_東京都、シン東京2050、ブロードリスニングによる政策転換.md": "05_tokyo_shin_tokyo_2050_policy_shift.md",
    "06_国政選挙でのブロードリスニングの利用.md": "06_broad_listening_in_national_elections.md",
    "06_01_チームみらい.md": "06_01_team_mirai.md",
    "06_02_チームみらい2026年衆院選.md": "06_02_team_mirai_2026_house_of_representatives_election.md",
    "06_03_日本維新の会.md": "06_03_nippon_ishin_no_kai.md",
    "06_04_国民民主党.md": "06_04_democratic_party_for_the_people.md",
    "06_05_公明党.md": "06_05_komeito.md",
    "07_地方選挙での活用.md": "07_use_in_local_elections.md",
    "07_02_いでい良輔氏のケース.md": "07_02_ryosuke_idei_case.md",
    "07_03_再生の道_尾花山和哉.md": "07_03_saisei_no_michi_kazuya_obanayama.md",
    "08_地方自治体での活用.md": "08_use_in_local_governments.md",
    "08_01_群馬県太田市の自分ごと化会議.md": "08_01_ota_city_jibungotoka_meeting.md",
    "08_02_広島県の事例.md": "08_02_hiroshima_case.md",
    "09_企業・NPOでの活用.md": "09_use_in_companies_and_npos.md",
    "09_01_アルティウスリンク_取材記事.md": "09_01_altius_link_interview.md",
    "09_02_サイボウズ.md": "09_02_cybozu.md",
    "10_ビジネスになったブロードリスニング.md": "10_broad_listening_as_a_business.md",
    "10_00_DD2030による広聴AIの開発活動.md": "10_00_dd2030_kocho_ai_development.md",
    "10_01_株式会社ブーツ.md": "10_01_boots_inc.md",
    "10_02_Code for Japan.md": "10_02_code_for_japan.md",
    "10_03_多元現実.md": "10_03_plural_reality.md",
    "10_04_Democracy X.md": "10_04_democracy_x.md",
    "10_05_litela_田中魁.md": "10_05_litela_kai_tanaka.md",
    "11_海外におけるブロードリスニングの流れ.md": "11_global_broad_listening_trends.md",
    "11_01_台湾.md": "11_01_taiwan.md",
    "11_02_Polisの誕生.md": "11_02_birth_of_polis.md",
    "11_03_ボーリンググリーン.md": "11_03_bowling_green.md",
    "11_04_イスラエルパレスチナRemesh事例.md": "11_04_israel_palestine_remesh_case.md",
    "11_05_Connective_Actionを力に変える.md": "11_05_harnessing_connective_action.md",
    "12_ブロードリスニング要素技術解説.md": "12_broad_listening_component_technologies.md",
    "13_広聴AIの技術スタック解説.md": "13_kocho_ai_tech_stack.md",
    "99_付録_公開事例一覧.md": "99_appendix_public_cases.md",
    "column/ワードクラウドは分析ではない.md": "column/word_cloud_is_not_analysis.md",
    "column/安全な共感.md": "column/safe_empathy.md",
    "column/レビューで試す広聴AI.md": "column/testing_kocho_ai_through_review.md",
    "column/1万件の声を集めて気づいたこと.md": "column/lessons_from_collecting_10000_voices.md",
    "column/パブリックコメントと向き合う大企業AI_富士通の取り組み.md": "column/fujitsu_enterprise_ai_for_public_comments.md",
}

TRANSLATION_SYSTEM_PROMPT = """You translate Japanese manuscript markdown into natural, publication-quality English.

Hard requirements:
- Preserve markdown structure.
- Preserve heading levels, lists, tables, links, code fences, and footnote anchors.
- Keep image paths and URLs unchanged.
- Translate image alt text and normal prose.
- Do not add first-use gloss expansions inside headings. Keep headings concise and natural;
  apply glosses only in body text.
- Preserve original author/byline credit in each chapter.
- If a chapter has an explicit author/byline line near the top, add a separate line
  immediately below it: "English translation by Luke Closs".
- Do not add repeated fault-disclaimer text in every chapter; keep such a note only where
  already present in the manuscript front matter.
- Return only translated markdown with no preface or code fences.

Terminology localization preferences (for a Western audience):
- For マニフェスト and related policy-document usage, prefer "platform" wording such as
  "electoral platform" or "party platform" based on context.
- Avoid "manifest" or "manifesto" unless it is an official quoted title that must be preserved.
- For 国会, prefer "Japanese Parliament".
- Avoid defaulting to "National Diet" unless needed for first-mention clarification. If needed,
  use "Japanese Parliament (the National Diet)" once, then continue with "Japanese Parliament".
- Prefer plain wording like "governor election" over "gubernatorial election".
- For チームみらい / Team Mirai:
  on first mention in each chapter, use
  "Team Mirai (referred to as 'Team Future' in this book)".
  After that, use "Team Future".
- For いどばた / Idobata:
  on first mention in each chapter, use
  "Watercooler (Idobata)", then continue with "Watercooler".
- For 会議 / kaigi, prefer "meeting" unless context clearly requires
  a different common English term (e.g., workshop, session, conference).
- For しゃべれるマニフェスト, prefer "interactive policy platform".
- Use official English party/organization names where available.
  Example: 日本維新の会 should be translated as
  "Japan Innovation Party (Nippon Ishin no Kai)" on first mention,
  then "Japan Innovation Party" thereafter.

Term-handling rules for the English manuscript:
- Baisoku Kaigi:
  keep as a product/brand name. On first mention in a chapter or section, use
  "Baisoku Kaigi, an AI-supported 'fast-track meeting' tool".
  On later mentions, use "Baisoku Kaigi".
- My Number:
  on first mention in a chapter or section, use
  "My Number, Japan's national ID system".
  On later mentions, use "My Number".
- Jibungotokaigi:
  normalize inconsistent spellings to "Jibungotokaigi".
  On first mention in a chapter or section, use
  "Jibungotokaigi, a citizen meeting designed to make public issues feel personally relevant".
  On later mentions, use "Jibungotokaigi".
- Shin Tokyo 2050:
  keep the original name as branding.
  On first mention in a chapter or section, prefer
  "Shin Tokyo 2050 ('New Tokyo 2050')".
  On later mentions, use "Shin Tokyo 2050".
- jigyo shiwake:
  keep and gloss on first mention as
  "jigyo shiwake, Japan's administrative budget-screening process".
  Later mentions: "jigyo shiwake".
- Nagatacho:
  keep and gloss on first mention as
  "Nagatacho, Japan's political center".
  Later mentions: "Nagatacho".
- shunto:
  prefer non-macron spelling "shunto".
  On first mention in a chapter or section, use
  "shunto, Japan's annual spring wage negotiations".
  Later mentions: "shunto".
- gongmin huiyi:
  on first mention, use
  "gongmin huiyi ('citizen conference')".
  Later mentions may use "gongmin huiyi" or "citizen conference" based on context.
- Mynaportal:
  keep and gloss on first mention as
  "Mynaportal, Japan's online government services portal".
  Later mentions: "Mynaportal".
- Reiwa 8:
  default to Gregorian year as "2026".
  Use "Reiwa 8 (2026)" only when the era notation itself is contextually important.

Global style behavior for these terms:
- Keep official product, platform, and project names in original form when they function as brands.
- Add short plain-English glosses on first mention for culture-specific Japanese or Taiwanese terms.
- Prefer functional explanations over literal translations when literal wording is awkward in English.
- Convert Japanese era years to Gregorian years by default.
- Normalize inconsistent spellings before applying first-use/later-use behavior.
- After first-use glossing, use the shorter later-use form consistently.
"""


def load_order_sections(order_file: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    with open(order_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            section_match = re.match(r"^\[(.+)\]$", line)
            if section_match:
                current_section = section_match.group(1)
                sections.setdefault(current_section, [])
                continue

            if current_section is None:
                continue

            sections[current_section].append(line)

    return sections


def map_relative_path(relative_path: str, path_map: dict[str, str]) -> str:
    return path_map.get(relative_path, relative_path)


def load_path_map(en_root: Path) -> dict[str, str]:
    path_map = dict(DEFAULT_PATH_MAP)
    path_map_file = en_root / "metadata" / "path_map.json"
    if path_map_file.exists():
        user_map = json.loads(path_map_file.read_text(encoding="utf-8"))
        path_map.update(user_map)
    return path_map


def save_path_map(en_root: Path, path_map: dict[str, str]) -> None:
    path_map_file = en_root / "metadata" / "path_map.json"
    path_map_file.parent.mkdir(parents=True, exist_ok=True)
    path_map_file.write_text(
        json.dumps(path_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_mapped_order_text(source_order_file: Path, path_map: dict[str, str]) -> str:
    mapped_lines = []
    with open(source_order_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or re.match(r"^\[.+\]$", stripped):
                mapped_lines.append(raw_line)
                continue

            mapped_lines.append(raw_line.replace(stripped, map_relative_path(stripped, path_map), 1))

    return "".join(mapped_lines)


def ensure_en_layout(
    source_order_file: Path,
    en_root: Path,
    overwrite_order: bool,
    path_map: dict[str, str],
) -> None:
    en_root.mkdir(parents=True, exist_ok=True)
    (en_root / "metadata").mkdir(parents=True, exist_ok=True)
    save_path_map(en_root, path_map)

    en_order_file = en_root / "book_order.txt"
    if overwrite_order or not en_order_file.exists():
        en_order_file.write_text(
            render_mapped_order_text(source_order_file, path_map),
            encoding="utf-8",
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"entries": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clean_model_output(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip() + "\n"


def translate_markdown_with_openai(
    source_markdown: str,
    model: str,
    temperature: float,
) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openai package is not installed. Install project dependencies first."
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": source_markdown},
        ],
    )
    content = response.choices[0].message.content or ""
    return clean_model_output(content)


def translate_file(
    source_path: Path,
    target_path: Path,
    model: str,
    temperature: float,
    dry_run: bool,
) -> tuple[str, str]:
    source_text = source_path.read_text(encoding="utf-8")
    source_hash = sha256_text(source_text)

    if dry_run:
        return source_hash, ""

    translated = translate_markdown_with_openai(
        source_markdown=source_text,
        model=model,
        temperature=temperature,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(translated, encoding="utf-8")
    return source_hash, sha256_text(translated)


def cmd_list_sections(args: argparse.Namespace) -> int:
    sections = load_order_sections(args.order_file)
    if not sections:
        print(f"No sections found in {args.order_file}")
        return 1

    print("Sections:")
    for name, files in sections.items():
        print(f"- {name} ({len(files)} files)")
    return 0


def resolve_section_name(sections: dict[str, list[str]], section: str | None, first: bool) -> str:
    if section:
        if section not in sections:
            valid = ", ".join(sections.keys())
            raise ValueError(f"Section '{section}' not found. Valid sections: {valid}")
        return section

    if first:
        return next(iter(sections))

    raise ValueError("Specify --section <name> or pass --first.")


def cmd_translate_section(args: argparse.Namespace) -> int:
    sections = load_order_sections(args.order_file)
    if not sections:
        print(f"No sections found in {args.order_file}")
        return 1

    try:
        section_name = resolve_section_name(sections, args.section, args.first)
    except ValueError as exc:
        print(str(exc))
        return 1

    manifest_path = args.en_root / "metadata" / "manifest.json"
    path_map = load_path_map(args.en_root)
    if not args.dry_run:
        ensure_en_layout(args.order_file, args.en_root, args.overwrite_order, path_map)
        manifest = load_manifest(manifest_path)
    else:
        manifest = {"entries": {}}
    entries = manifest.setdefault("entries", {})

    files = sections[section_name]
    print(f"Translating section [{section_name}] with {len(files)} files")

    translated_count = 0
    skipped_count = 0

    for relative_path in files:
        source_path = args.source_root / relative_path
        target_relative_path = map_relative_path(relative_path, path_map)
        target_path = args.en_root / target_relative_path

        if not source_path.exists():
            print(f"[WARN] Missing source file: {source_path}")
            continue

        if target_path.exists() and not args.overwrite:
            print(f"[SKIP] {relative_path} (already exists, use --overwrite to replace)")
            skipped_count += 1
            continue

        print(f"[TRANSLATE] {relative_path}")
        try:
            source_hash, target_hash = translate_file(
                source_path=source_path,
                target_path=target_path,
                model=args.model,
                temperature=args.temperature,
                dry_run=args.dry_run,
            )
        except RuntimeError as exc:
            print(f"[ERROR] {relative_path}: {exc}")
            return 1

        if args.dry_run:
            print(f"[DRY-RUN] Would write: {target_path}")
        else:
            translated_count += 1

        entries[relative_path] = {
            "source_path": relative_path,
            "target_path": target_relative_path,
            "source_hash": source_hash,
            "target_hash": target_hash,
            "section": section_name,
            "model": args.model,
            "status": "translated" if not args.dry_run else "dry-run",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    manifest["last_section"] = section_name
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    print(
        f"Done. translated={translated_count}, skipped={skipped_count}, "
        f"dry_run={'yes' if args.dry_run else 'no'}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    def add_common_args(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--source-root",
            type=Path,
            default=Path(__file__).parent.parent,
            help="Repository root that contains Japanese source files",
        )
        target.add_argument(
            "--order-file",
            type=Path,
            default=Path(__file__).parent.parent / "book_order.txt",
            help="Order file used to define sections",
        )
        target.add_argument(
            "--en-root",
            type=Path,
            default=Path(__file__).parent.parent / "en",
            help="Output root for English markdown",
        )

    parser = argparse.ArgumentParser(description="English translation tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-sections", help="List section names from the order file")
    add_common_args(list_parser)

    translate_parser = subparsers.add_parser(
        "translate-section",
        help="Translate a section from Japanese markdown to English markdown",
    )
    add_common_args(translate_parser)
    translate_parser.add_argument("--section", help="Section name from [section] in book_order.txt")
    translate_parser.add_argument(
        "--first",
        action="store_true",
        help="Translate the first section from the order file",
    )
    translate_parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="OpenAI model for translation",
    )
    translate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for translation",
    )
    translate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translated files",
    )
    translate_parser.add_argument(
        "--overwrite-order",
        action="store_true",
        help="Overwrite en/book_order.txt from source order file",
    )
    translate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan translation without writing files or calling the API",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-sections":
        exit_code = cmd_list_sections(args)
    elif args.command == "translate-section":
        exit_code = cmd_translate_section(args)
    else:
        print(f"Unknown command: {args.command}")
        exit_code = 1

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
