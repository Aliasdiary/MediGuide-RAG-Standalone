"""Download, validate, and prepare a deterministic MedQuAD subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
ALLOWED_SOURCE_DIRS = {
    "1_CancerGov_QA",
    "2_GARD_QA",
    "3_GHR_QA",
    "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "7_SeniorHealth_QA",
    "8_NHLBI_QA_XML",
    "9_CDC_QA",
}
LICENSE = "CC BY 4.0"


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def first_text(root: ET.Element, paths: Iterable[str]) -> str:
    for path in paths:
        node = root.find(path)
        if node is not None:
            text = clean_text("".join(node.itertext()))
            if text:
                return text
    return ""


def source_directory(xml_path: Path) -> str:
    for part in xml_path.parts:
        if part in ALLOWED_SOURCE_DIRS:
            return part
    return ""


def parse_xml(xml_path: Path) -> List[Dict[str, str]]:
    root = ET.parse(xml_path).getroot()
    source_dir = source_directory(xml_path)
    if not source_dir:
        return []

    source_org = clean_text(root.attrib.get("source")) or source_dir.split("_", 1)[-1]
    source_url = clean_text(root.attrib.get("url"))
    focus = first_text(root, ["./Focus"])
    umls_cui = first_text(root, ["./FocusAnnotations/UMLS/CUIs/CUI"])
    semantic_type = first_text(root, ["./FocusAnnotations/UMLS/SemanticTypes/SemanticType"])
    records = []

    for pair in root.findall(".//QAPair"):
        question_node = pair.find("./Question")
        answer_node = pair.find("./Answer")
        question = clean_text("".join(question_node.itertext())) if question_node is not None else ""
        answer = clean_text("".join(answer_node.itertext())) if answer_node is not None else ""
        if not question or not answer:
            continue
        records.append(
            {
                "id": clean_text(question_node.attrib.get("qid")) or clean_text(pair.attrib.get("pid")),
                "question": question,
                "answer": answer,
                "question_type": clean_text(question_node.attrib.get("qtype")).lower() or "information",
                "focus": focus,
                "source_org": source_org,
                "source_url": source_url,
                "umls_cui": umls_cui,
                "semantic_type": semantic_type,
                "source_subset": source_dir,
                "license": LICENSE,
            }
        )
    return records


def stratified_sample(records: List[Dict[str, str]], limit: int, seed: int) -> List[Dict[str, str]]:
    if limit <= 0 or limit >= len(records):
        return sorted(records, key=lambda item: item["id"])

    rng = random.Random(seed)
    buckets: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        buckets[(record["source_org"], record["question_type"])].append(record)
    queues = []
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
        queues.append(deque(buckets[key]))

    selected = []
    while queues and len(selected) < limit:
        active = []
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
            if queue:
                active.append(queue)
        queues = active
    return selected


def download_archive(url: str, destination: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "MediGuide-RAG/1.0"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if not zipfile.is_zipfile(destination):
        raise ValueError("Downloaded MedQuAD archive is not a valid ZIP file")
    return digest.hexdigest()


def prepare(url: str, output: Path, limit: int, seed: int) -> Dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="medquad_") as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / "medquad.zip"
        archive_sha256 = download_archive(url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            if not any(name.endswith("LICENSE.txt") for name in names):
                raise ValueError("MedQuAD archive does not contain LICENSE.txt")
            archive.extractall(temp_root / "extracted")

        records = []
        parse_errors = []
        for xml_path in sorted((temp_root / "extracted").rglob("*.xml")):
            if not source_directory(xml_path):
                continue
            try:
                records.extend(parse_xml(xml_path))
            except (ET.ParseError, OSError) as exc:
                parse_errors.append(f"{xml_path.name}: {exc}")

        if not records:
            raise ValueError("No valid MedQuAD question-answer records were parsed")
        selected = stratified_sample(records, limit, seed)

        temp_output = output.with_suffix(output.suffix + ".tmp")
        with temp_output.open("w", encoding="utf-8", newline="\n") as stream:
            for record in selected:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        shutil.move(str(temp_output), output)

    data_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "dataset": "MedQuAD",
        "dataset_version": "master",
        "source_url": url,
        "license": LICENSE,
        "allowed_source_subsets": sorted(ALLOWED_SOURCE_DIRS),
        "excluded_subsets": [
            "10_MPlus_ADAM_QA",
            "11_MPlusDrugs_QA",
            "12_MPlusHerbsSupplements_QA",
        ],
        "available_records": len(records),
        "selected_records": len(selected),
        "seed": seed,
        "archive_sha256": archive_sha256,
        "data_sha256": data_sha256,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def default_output() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "medquad_5000.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="MedQuAD ZIP archive URL")
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.url, args.output.resolve(), args.limit, args.seed)
    print(
        f"Prepared {manifest['selected_records']} MedQuAD records at {args.output.resolve()}\n"
        f"Data SHA256: {manifest['data_sha256']}"
    )


if __name__ == "__main__":
    main()
