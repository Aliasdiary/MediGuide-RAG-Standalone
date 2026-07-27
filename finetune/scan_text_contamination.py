"""Scan SFT data or generations for template/copyright/disclaimer contamination."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List


PATTERNS = [
    r"本回答可能包含",
    r"不能替代完整医学诊断",
    r"仅供参考",
    r"不可替代医生",
    r"不能替代医生诊断",
    r"未经授权",
    r"不得用于商业",
    r"copyright",
    r"source:",
    r"final answer in chinese",
]


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.jsonl")
            yield from path.rglob("*.txt")
            yield from path.rglob("*.md")


def scan_file(path: Path, regexes: List[re.Pattern], max_hits: int) -> int:
    hits = 0
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line_no, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                text = "\n".join(str(payload.get(key, "")) for key in ("instruction", "input", "output", "answer"))
            except json.JSONDecodeError:
                pass
            for regex in regexes:
                if regex.search(text):
                    print(f"{path}:{line_no}: {regex.pattern}")
                    hits += 1
                    break
            if hits >= max_hits:
                break
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan files for common SFT text contamination patterns.")
    parser.add_argument("paths", nargs="*", default=["finetune", "data"])
    parser.add_argument("--max-hits-per-file", type=int, default=20)
    args = parser.parse_args()

    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS]
    total = 0
    for path in iter_files(Path(item) for item in args.paths):
        total += scan_file(path, regexes, args.max_hits_per_file)
    print(f"total_hits={total}")


if __name__ == "__main__":
    main()

