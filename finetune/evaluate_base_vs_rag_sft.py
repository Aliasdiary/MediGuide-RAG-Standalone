"""Compare Base Qwen2.5-Instruct with the unified MediGuide RAG-SFT path."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, MediGuideConfig  # noqa: E402
from finetune.evaluate_rag_sft import (  # noqa: E402
    DEFAULT_QUESTIONS,
    aggregate,
    build_runtime,
    generate_rag_sft_answer,
    load_jsonl,
    load_model,
    score_answer,
)
from finetune.infer_rag_sft import normalize_answer, run_generation  # noqa: E402
from finetune.infer_sft import SAFETY_SYSTEM_PROMPT, build_qwen_chatml_prompt  # noqa: E402


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
DEFAULT_RAG_SFT_MODEL = "/root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export/qwen25-3b-mediguide-sft"
DEFAULT_OUTPUT_DIR = "finetune/eval_results_compare"

COMPARABLE_KEYS = [
    "safety_compliance",
    "keyword_coverage",
    "medication_safety",
    "emergency_awareness",
    "chinese_output",
    "hallucination_control",
]

RAG_ONLY_KEYS = [
    "evidence_gate_pass",
    "retrieval_has_hits",
]


def generate_base_answer(case: Dict[str, Any], args, tokenizer, model, device: str) -> str:
    prompt = build_qwen_chatml_prompt(case["question"], SAFETY_SYSTEM_PROMPT)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "num_beams": 1,
    }
    return normalize_answer(run_generation(model, tokenizer, device, prompt, generation_kwargs))


def aggregate_comparable(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    result = aggregate(rows)
    result["overall_comparable"] = round(sum(result[key] for key in COMPARABLE_KEYS) / len(COMPARABLE_KEYS), 4)
    return result


def write_report(base_metrics: Dict[str, float], rag_metrics: Dict[str, float], output_path: Path) -> None:
    lines = [
        "# Base vs MediGuide RAG-SFT Evaluation Report",
        "",
        "This report compares Base Qwen2.5-3B-Instruct without RAG against the unified Retrieval-Aware RAG-SFT path.",
        "The metrics are engineering checks, not clinical diagnosis accuracy.",
        "",
        "| Comparable Metric | Base | RAG-SFT | Gain |",
        "|---|---:|---:|---:|",
    ]
    for key in COMPARABLE_KEYS + ["overall_comparable"]:
        base_value = base_metrics.get(key, 0.0)
        rag_value = rag_metrics.get(key, 0.0)
        lines.append(f"| {key} | {base_value:.1%} | {rag_value:.1%} | {rag_value - base_value:+.1%} |")

    lines.extend(["", "| RAG-SFT Chain Metric | Score |", "|---|---:|"])
    for key in RAG_ONLY_KEYS:
        lines.append(f"| {key} | {rag_metrics.get(key, 0.0):.1%} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def release_model(tokenizer, model) -> None:
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Base Qwen2.5 with MediGuide RAG-SFT.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--rag-sft-model-path", default=DEFAULT_RAG_SFT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_CONFIG.embedding_model)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--max-context-chars", type=int, default=2400)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cases = load_jsonl(args.questions, args.limit)

    print("[load] base model")
    base_tokenizer, base_model = load_model(args.base_model_path, args.device)
    base_rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[base] {index}/{len(cases)} {case['id']}")
        answer = generate_base_answer(case, args, base_tokenizer, base_model, args.device)
        base_rows.append(
            {
                "id": case["id"],
                "mode": "base",
                "category": case.get("category", ""),
                "question": case["question"],
                "answer": answer,
                "scores": score_answer(answer, case, "not_applicable", []),
            }
        )
    release_model(base_tokenizer, base_model)

    config = MediGuideConfig.from_dict(DEFAULT_CONFIG.to_dict())
    config.embedding_model = args.embedding_model
    data_module, retrieval_module, gate = build_runtime(config)

    print("[load] rag-sft model")
    rag_tokenizer, rag_model = load_model(args.rag_sft_model_path, args.device)
    rag_rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[rag-sft] {index}/{len(cases)} {case['id']}")
        answer, evidence_status, retrieval_query, hits = generate_rag_sft_answer(
            case, args, data_module, retrieval_module, gate, rag_tokenizer, rag_model, args.device
        )
        rag_rows.append(
            {
                "id": case["id"],
                "mode": "rag_sft",
                "category": case.get("category", ""),
                "question": case["question"],
                "retrieval_query": retrieval_query,
                "evidence_status": evidence_status,
                "retrieval_hits": hits,
                "answer": answer,
                "scores": score_answer(answer, case, evidence_status, hits),
            }
        )
    release_model(rag_tokenizer, rag_model)

    base_metrics = aggregate_comparable(base_rows)
    rag_metrics = aggregate_comparable(rag_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "base_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in base_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "rag_sft_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rag_rows) + "\n",
        encoding="utf-8",
    )
    metrics = {"base": base_metrics, "rag_sft": rag_metrics}
    (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(base_metrics, rag_metrics, output_dir / "comparison_report.md")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

