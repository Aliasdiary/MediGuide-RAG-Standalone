"""Build MediGuide SFT data from prepared MedQuAD JSONL.

The output format is Alpaca-style JSONL and can be consumed by LLaMA-Factory.
This script intentionally keeps the original medical facts from MedQuAD instead
of rewriting them with a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


QUESTION_TYPE_TASKS = {
    "information": "medical education",
    "causes": "cause explanation",
    "symptoms": "symptom education",
    "treatment": "treatment education",
    "exams and tests": "exam and test education",
    "prevention": "prevention education",
    "susceptibility": "risk-factor education",
    "considerations": "medical safety guidance",
    "diagnosis": "exam and diagnosis education",
    "management": "disease management education",
}

SAFETY_CASES = [
    {
        "instruction": "请用中文回答医疗安全问题，并遵守不诊断、不处方、不提供个体化剂量的边界。",
        "input": "用户问题：我能不能直接把降压药剂量加倍？\n\n可用资料：当前资料未提供针对个人用药剂量调整的依据。",
        "output": "不建议自行把降压药剂量加倍。药物剂量需要结合血压水平、合并疾病、肝肾功能和正在使用的其他药物综合判断，应由医生或药师评估后调整。当前资料不足以支持个体化剂量建议。本回答仅用于健康科普，不能替代医生诊断或处方。",
    },
    {
        "instruction": "请用中文回答医疗安全问题，并在危险信号出现时优先提示急诊。",
        "input": "用户问题：突然胸痛并且呼吸困难，我应该等明天再看吗？\n\n可用资料：胸痛、呼吸困难可能提示需要紧急医疗评估。",
        "output": "不要等到明天。突然胸痛并伴随呼吸困难属于危险信号，应立即联系当地急救服务或前往急诊。在线回答不能判断具体病因，也不能替代急救评估。本回答仅用于健康科普，不能替代医生诊断。",
    },
    {
        "instruction": "请用中文回答医疗安全问题，并遵守不诊断、不处方、不提供个体化剂量的边界。",
        "input": "用户问题：抗生素可以自己按成人剂量吃两天吗？\n\n可用资料：当前资料不足以支持个人抗生素剂量和疗程建议。",
        "output": "不建议自行按成人剂量使用抗生素。抗生素是否需要使用、用哪一种、剂量和疗程都需要医生或药师根据感染类型、病原体和个人情况判断。自行用药可能导致治疗不充分、不良反应或耐药风险。本回答仅用于健康科普，不能替代医生诊断或处方。",
    },
    {
        "instruction": "请用中文回答医疗安全问题，并在危险信号出现时优先提示急诊。",
        "input": "用户问题：孩子发烧抽搐并且意识不清，应该怎么办？\n\n可用资料：发热伴抽搐、意识不清属于需要紧急处理的危险信号。",
        "output": "请立即联系当地急救服务或前往急诊，不要等待在线回答。发烧、抽搐并伴随意识不清可能涉及严重感染、神经系统问题或其他急症，需要专业医疗人员现场评估。本回答仅用于健康科普，不能替代医生诊断或急救服务。",
    },
    {
        "instruction": "请用中文回答医疗健康问题。对无法依据资料回答的内容要明确说明信息不足。",
        "input": "用户问题：MediGuide 能告诉我今天北京哪家医院排队最短吗？\n\n可用资料：MedQuAD 医疗科普资料中不包含实时医院排队信息。",
        "output": "当前资料无法回答今天北京哪家医院排队最短。该问题需要实时医院挂号或排队数据，建议查询医院官方渠道、当地预约挂号平台或电话咨询医院。本回答仅用于健康科普信息说明，不能替代实时就医服务。",
    },
    {
        "instruction": "请用中文回答医疗安全问题，并拒绝不可靠治疗承诺。",
        "input": "用户问题：某某网红保健品能不能治好糖尿病？\n\n可用资料：当前资料不支持该保健品可以治愈糖尿病的说法。",
        "output": "不能根据当前资料认定某个网红保健品可以治好糖尿病。糖尿病管理通常需要正规医学评估、生活方式管理和必要时的药物治疗。不要用保健品替代医生建议的治疗方案。本回答仅用于健康科普，不能替代医生诊断或治疗建议。",
    },
]


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question = clean_text(record.get("question"))
            answer = clean_text(record.get("answer"))
            source_url = clean_text(record.get("source_url"))
            focus = clean_text(record.get("focus")) or question
            if not question or not answer or not source_url or not focus:
                continue
            record["question"] = question
            record["answer"] = answer
            record["source_url"] = source_url
            record["focus"] = focus
            record["question_type"] = clean_text(record.get("question_type")).lower() or "information"
            record["source_org"] = clean_text(record.get("source_org")) or "Unknown"
            records.append(record)
    return records


def record_key(record: Dict[str, Any]) -> str:
    raw = "|".join(
        [
            record.get("focus", ""),
            record.get("question_type", ""),
            record.get("question", ""),
            record.get("source_url", ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def filter_records(records: Iterable[Dict[str, Any]], max_answer_chars: int) -> List[Dict[str, Any]]:
    seen = set()
    filtered = []
    for record in records:
        if len(record["answer"]) > max_answer_chars:
            continue
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(record)
    return filtered


def medquad_to_sample(record: Dict[str, Any]) -> Dict[str, str]:
    question_type = record.get("question_type", "information")
    task = QUESTION_TYPE_TASKS.get(question_type, "medical education")
    instruction = (
        "Answer the medical education question using only the provided MedQuAD source. "
        "Do not provide a personal diagnosis, prescription, or individualized dosage. "
        "Include the source organization and URL."
    )
    input_text = (
        f"Task: {task}\n"
        f"Topic: {record['focus']}\n"
        f"Question type: {question_type}\n"
        f"User question: {record['question']}\n\n"
        f"MedQuAD source:\n"
        f"Organization: {record['source_org']}\n"
        f"URL: {record['source_url']}\n"
        f"Answer: {record['answer']}"
    )
    output = (
        f"{record['answer']}\n\n"
        f"Source: {record['source_org']}, {record['source_url']}\n"
        "This answer is for health education only and cannot replace professional medical diagnosis or treatment."
    )
    return {"instruction": instruction, "input": input_text, "output": output}


def split_samples(samples: List[Dict[str, str]], train_size: int, valid_size: int, seed: int):
    rng = random.Random(seed)
    rng.shuffle(samples)
    if train_size <= 0:
        train_size = max(len(samples) - valid_size, 0)
    train = samples[:train_size]
    valid = samples[train_size : train_size + valid_size]
    return train, valid


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_dataset_info(output_dir: Path) -> None:
    dataset_info = {
        "mediguide_sft_train": {
            "file_name": "mediguide_sft_train.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "mediguide_sft_valid": {
            "file_name": "mediguide_sft_valid.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/medquad_5000.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("finetune/llamafactory_data"))
    parser.add_argument("--train-size", type=int, default=4500)
    parser.add_argument("--valid-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-answer-chars", type=int, default=5000)
    parser.add_argument("--safety-repeat", type=int, default=30)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = filter_records(load_records(args.input.resolve()), args.max_answer_chars)
    samples = [medquad_to_sample(record) for record in records]
    for _ in range(max(args.safety_repeat, 0)):
        samples.extend(SAFETY_CASES)

    train, valid = split_samples(samples, args.train_size, args.valid_size, args.seed)
    train_count = write_jsonl(output_dir / "mediguide_sft_train.jsonl", train)
    valid_count = write_jsonl(output_dir / "mediguide_sft_valid.jsonl", valid)
    write_dataset_info(output_dir)

    type_counts = Counter(record["question_type"] for record in records)
    manifest = {
        "input": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "base_records_after_filter": len(records),
        "safety_cases": len(SAFETY_CASES) * max(args.safety_repeat, 0),
        "train_samples": train_count,
        "valid_samples": valid_count,
        "seed": args.seed,
        "max_answer_chars": args.max_answer_chars,
        "question_type_counts": dict(type_counts.most_common()),
        "format": "llamafactory-alpaca-jsonl",
    }
    (output_dir / "mediguide_sft_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
