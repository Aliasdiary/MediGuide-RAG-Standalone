"""Scan data files for disclaimer, copyright, and template contamination."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


PATTERNS = [
    "本回答可能包含",
    "不能替代完整医学诊断",
    "仅供参考",
    "不可替代医生",
    "不能替代医生",
    "免责声明",
    "版权声明",
    "版权所有",
    "未经许可",
    "商业用途",
    "本内容由",
    "本回答由",
    "based on your diagnosis",
    "health care provider will",
    "all rights reserved",
    "copyright",
]


def iter_files(paths: Iterable[str]):
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*")


def scan_file(path: Path, max_examples: int):
    if path.suffix.lower() not in {".jsonl", ".json", ".txt", ".md", ".yaml", ".yml"}:
        return {}

    counts = {pattern: 0 for pattern in PATTERNS}
    examples = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {}

    for line_no, line in enumerate(lines, start=1):
        text = line
        if path.suffix.lower() == ".jsonl":
            try:
                obj = json.loads(line)
                text = json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        lowered = text.casefold()
        matched = [pattern for pattern in PATTERNS if pattern.casefold() in lowered]
        if matched:
            for pattern in matched:
                counts[pattern] += 1
            if len(examples) < max_examples:
                snippet = re.sub(r"\s+", " ", text)[:240]
                examples.append((line_no, matched, snippet))

    total = sum(counts.values())
    if not total:
        return {}
    return {"total": total, "counts": counts, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan MediGuide data for text contamination.")
    parser.add_argument("paths", nargs="*", default=["finetune", "data"])
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    found = False
    for path in iter_files(args.paths):
        result = scan_file(path, args.max_examples)
        if not result:
            continue
        found = True
        print(f"\n{path}")
        print(f"total_matches={result['total']}")
        for pattern, count in result["counts"].items():
            if count:
                print(f"  {pattern}: {count}")
        for line_no, matched, snippet in result["examples"]:
            print(f"  line {line_no} | {', '.join(matched)} | {snippet}")

    if not found:
        print("No contamination patterns found.")


if __name__ == "__main__":
    main()
