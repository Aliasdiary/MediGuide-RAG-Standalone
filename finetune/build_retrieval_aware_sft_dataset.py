"""Build Retrieval-Aware SFT data from real MediGuide-RAG retrieval results."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, MediGuideConfig  # noqa: E402
from finetune.infer_rag_sft import format_evidence, rewrite_retrieval_query  # noqa: E402
from rag_modules import DataPreparationModule, EvidenceGate, IndexConstructionModule, RetrievalOptimizationModule  # noqa: E402


INSTRUCTION = (
    "你是医疗健康科普问答 Agent。请根据用户问题、候选医学证据和证据状态回答。"
    "必须先判断证据是否直接支持问题；证据不足或主体错配时应拒绝给出具体医学结论。"
)


SAFETY_REFUSALS = [
    {
        "question": "我能不能直接把降压药剂量加倍？",
        "query": "high blood pressure hypertension medication dose adjustment",
        "answer": "不建议自行把降压药剂量加倍。药物剂量需要结合血压水平、合并疾病、肝肾功能和正在使用的其他药物综合判断，应由医生或药师评估后调整。自行调整剂量可能带来血压控制异常或不良反应风险。",
    },
    {
        "question": "我被狗咬了怎么办？",
        "query": "dog bite animal bite human wound care rabies exposure post exposure prophylaxis tetanus medical care wash wound",
        "answer": "被狗咬后需要及时处理伤口并尽快就医评估狂犬病暴露和破伤风风险。若伤口较深、出血明显、动物来源不明或无法确认免疫情况，更应尽快联系医疗机构处理。",
    },
]


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_input(question: str, evidence_status: str, evidence_text: str) -> str:
    return f"""用户问题：
{question}

证据状态：
{evidence_status}

候选医学证据：
{evidence_text}"""


def compact_answer(text: str, max_chars: int = 900) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(".", 1)[0].strip()
    return cut if len(cut) >= 120 else text[:max_chars].strip()


def record_to_doc(record: dict) -> Document:
    content = f"# {record.get('focus') or record['question']}\n\n## Question\n{record['question']}\n\n## Answer\n{record['answer']}"
    return Document(
        page_content=content,
        metadata={
            "parent_id": record.get("parent_id") or f"{record.get('source_url')}|{record.get('id', record['question'])}",
            "focus": record.get("focus", ""),
            "question": record.get("question", ""),
            "question_type": record.get("question_type", ""),
            "source_org": record.get("source_org", ""),
            "source_url": record.get("source_url", ""),
            "semantic_type": record.get("semantic_type", ""),
        },
    )


def build_rag_components(config: MediGuideConfig):
    data_module = DataPreparationModule(config.data_path)
    chunks = data_module.chunk_documents() if data_module.documents else None
    if chunks is None:
        data_module.load_documents()
        chunks = data_module.chunk_documents()
    index_module = IndexConstructionModule(
        embedding_model=config.embedding_model,
        index_save_path=config.index_save_path,
        dataset_fingerprint=data_module.dataset_fingerprint(),
    )
    vectorstore = index_module.load_or_create_index(chunks)
    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    return data_module, retrieval_module


def make_retrieval_sample(record: dict, data_module, retrieval_module, gate: EvidenceGate, max_context_chars: int) -> dict:
    question = record["question"].strip()
    query = rewrite_retrieval_query(question)
    child_hits = retrieval_module.hybrid_search(query, top_k=gate.dynamic_top_k(question, 4))
    parent_hits = data_module.get_parent_documents(child_hits)

    oracle_doc = record_to_doc(record)
    if not any(doc.metadata.get("source_url") == oracle_doc.metadata.get("source_url") for doc in parent_hits):
        parent_hits = [oracle_doc] + parent_hits

    gate_result = gate.assess(question, parent_hits, route=gate.infer_route(question))
    evidence_text = format_evidence(gate_result.usable_docs, max_context_chars=max_context_chars)
    output = compact_answer(record["answer"])
    return {
        "instruction": INSTRUCTION,
        "input": build_input(question, gate_result.status, evidence_text),
        "output": output,
    }


def make_safety_sample(item: dict, data_module, retrieval_module, gate: EvidenceGate, max_context_chars: int) -> dict:
    child_hits = retrieval_module.hybrid_search(item["query"], top_k=gate.dynamic_top_k(item["question"], 4))
    parent_hits = data_module.get_parent_documents(child_hits)
    gate_result = gate.assess(item["question"], parent_hits, route=gate.infer_route(item["question"]))
    evidence_text = format_evidence(gate_result.usable_docs, max_context_chars=max_context_chars)
    return {
        "instruction": INSTRUCTION,
        "input": build_input(item["question"], gate_result.status, evidence_text),
        "output": item["answer"],
    }


def merge_base_sft(output_rows: List[dict], base_dir: Path, limit: int) -> List[dict]:
    train_path = base_dir / "mediguide_sft_train.jsonl"
    if limit <= 0 or not train_path.exists():
        return output_rows
    base_rows = load_jsonl(train_path)
    return output_rows + base_rows[:limit]


def write_dataset_info(output_dir: Path) -> None:
    dataset_info = {
        "mediguide_retrieval_sft_train": {
            "file_name": "mediguide_retrieval_sft_train.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "mediguide_retrieval_sft_valid": {
            "file_name": "mediguide_retrieval_sft_valid.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Retrieval-Aware SFT data.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "medquad_5000.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "finetune" / "retrieval_aware_data"))
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--valid-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default=DEFAULT_CONFIG.embedding_model)
    parser.add_argument("--max-context-chars", type=int, default=1800)
    parser.add_argument("--base-sft-dir", default=str(PROJECT_ROOT / "finetune" / "llamafactory_data"))
    parser.add_argument("--base-sft-mix", type=int, default=500)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(Path(args.input))
    rng.shuffle(records)

    config = MediGuideConfig.from_dict(DEFAULT_CONFIG.to_dict())
    config.embedding_model = args.embedding_model
    data_module, retrieval_module = build_rag_components(config)
    gate = EvidenceGate(min_support=config.evidence_gate_min_score, max_docs=config.evidence_gate_max_docs)

    target_total = args.train_size + args.valid_size
    selected = records[: max(target_total, 1)]
    rows = [
        make_retrieval_sample(record, data_module, retrieval_module, gate, args.max_context_chars)
        for record in selected
    ]
    rows.extend(make_safety_sample(item, data_module, retrieval_module, gate, args.max_context_chars) for item in SAFETY_REFUSALS)

    train_rows = rows[: args.train_size]
    train_rows = merge_base_sft(train_rows, Path(args.base_sft_dir), args.base_sft_mix)
    valid_rows = rows[args.train_size : args.train_size + args.valid_size]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_count = save_jsonl(output_dir / "mediguide_retrieval_sft_train.jsonl", train_rows)
    valid_count = save_jsonl(output_dir / "mediguide_retrieval_sft_valid.jsonl", valid_rows)
    write_dataset_info(output_dir)

    manifest = {
        "source": str(args.input),
        "train_count": train_count,
        "valid_count": valid_count,
        "seed": args.seed,
        "embedding_model": args.embedding_model,
        "max_context_chars": args.max_context_chars,
        "base_sft_mix": args.base_sft_mix,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

