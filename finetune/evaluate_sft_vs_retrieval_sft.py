"""Compare plain SFT + fixed RAG evidence against Retrieval-Aware SFT.

This is the fair comparison for the current project:

same question
same retrieved evidence
same Evidence Gate status
same prompt
same generation parameters
only the generator model changes
"""

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
    load_jsonl,
    load_model,
    score_answer,
)
from finetune.infer_rag_sft import (  # noqa: E402
    SYSTEM_PROMPT,
    build_chat_prompt,
    build_rewrite_prompt,
    build_user_prompt,
    chinese_safety_fallback,
    format_evidence,
    mostly_english,
    normalize_answer,
    rewrite_retrieval_query,
    run_generation,
)


DEFAULT_BASELINE_SFT_MODEL = "/root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export/qwen25-3b-baseline-sft"
DEFAULT_RETRIEVAL_SFT_MODEL = "/root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export/qwen25-3b-mediguide-sft"
DEFAULT_OUTPUT_DIR = "finetune/eval_results_sft_compare"

LOWER_IS_BETTER = {"unsupported_fact_rate", "wrong_evidence_influence_rate", "wrong_force_answer_rate"}
MODEL_METRICS = [
    "safety_compliance",
    "factual_support_rate",
    "unsupported_fact_rate",
    "wrong_evidence_influence_rate",
    "subject_consistency",
    "unsupported_refusal_correct",
    "wrong_force_answer_rate",
    "medication_safety",
    "emergency_recall",
    "chinese_output",
    "template_leakage_control",
    "overall",
]
CHAIN_METRICS = ["evidence_gate_pass", "retrieval_has_hits"]


def precompute_evidence(cases: List[Dict[str, Any]], args, data_module, retrieval_module, gate) -> List[Dict[str, Any]]:
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[retrieve] {index}/{len(cases)} {case['id']}")
        question = case["question"]
        retrieval_query = rewrite_retrieval_query(question)
        child_hits = retrieval_module.hybrid_search(retrieval_query, top_k=gate.dynamic_top_k(question, args.top_k))
        parent_hits = data_module.get_parent_documents(child_hits)
        gate_result = gate.assess(question, parent_hits, route=gate.infer_route(question))
        evidence_text = format_evidence(gate_result.usable_docs, args.max_context_chars)
        hits = [
            {
                "focus": doc.metadata.get("focus", ""),
                "question_type": doc.metadata.get("question_type", ""),
                "source_org": doc.metadata.get("source_org", ""),
                "source_url": doc.metadata.get("source_url", ""),
                "evidence_status": doc.metadata.get("evidence_status", ""),
                "evidence_support": doc.metadata.get("evidence_support", 0),
            }
            for doc in parent_hits
        ]
        rows.append(
            {
                "case": case,
                "retrieval_query": retrieval_query,
                "evidence_status": gate_result.status,
                "evidence_text": evidence_text,
                "retrieval_hits": hits,
            }
        )
    return rows


def generate_with_fixed_evidence(item: Dict[str, Any], args, tokenizer, model, device: str) -> str:
    case = item["case"]
    prompt = build_chat_prompt(
        tokenizer,
        SYSTEM_PROMPT,
        build_user_prompt(case["question"], item["evidence_status"], item["evidence_text"]),
    )
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "num_beams": 1,
    }
    answer = normalize_answer(run_generation(model, tokenizer, device, prompt, generation_kwargs))
    if mostly_english(answer):
        rewrite_prompt = build_chat_prompt(
            tokenizer,
            "请将候选回答改写为简体中文，保留医学科普信息，不新增诊断、处方或个体化剂量建议。",
            build_rewrite_prompt(case["question"], answer),
        )
        answer = normalize_answer(run_generation(model, tokenizer, device, rewrite_prompt, generation_kwargs))
        if mostly_english(answer):
            answer = chinese_safety_fallback(case["question"], item["evidence_status"], item["evidence_text"])
    return answer


def evaluate_model(label: str, model_path: str, evidence_items: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    print(f"[load] {label}: {model_path}")
    tokenizer, model = load_model(model_path, args.device)
    rows = []
    for index, item in enumerate(evidence_items, start=1):
        case = item["case"]
        print(f"[{label}] {index}/{len(evidence_items)} {case['id']}")
        answer = generate_with_fixed_evidence(item, args, tokenizer, model, args.device)
        rows.append(
            {
                "id": case["id"],
                "mode": label,
                "category": case.get("category", ""),
                "question": case["question"],
                "retrieval_query": item["retrieval_query"],
                "evidence_status": item["evidence_status"],
                "retrieval_hits": item["retrieval_hits"],
                "evidence_text": item["evidence_text"],
                "answer": answer,
                "scores": score_answer(
                    answer,
                    case,
                    item["evidence_status"],
                    item["retrieval_hits"],
                    item["evidence_text"],
                ),
            }
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def metric_improvement(metric: str, baseline: float, retrieval_aware: float) -> float:
    if metric in LOWER_IS_BETTER:
        return baseline - retrieval_aware
    return retrieval_aware - baseline


def write_report(baseline_metrics: Dict[str, float], retrieval_metrics: Dict[str, float], output_path: Path) -> None:
    lines = [
        "# Plain SFT + RAG Evidence vs Retrieval-Aware SFT",
        "",
        "This report uses fixed RAG evidence for both models. Only the generator model changes.",
        "The metrics are engineering checks, not clinical diagnosis accuracy.",
        "",
        "| Metric | Plain SFT + RAG | Retrieval-Aware SFT | Improvement | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    for metric in MODEL_METRICS:
        baseline = baseline_metrics.get(metric, 0.0)
        retrieval = retrieval_metrics.get(metric, 0.0)
        direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        improvement = metric_improvement(metric, baseline, retrieval)
        lines.append(f"| {metric} | {baseline:.1%} | {retrieval:.1%} | {improvement:+.1%} | {direction} |")

    lines.extend(["", "## Shared Retrieval Chain", "", "| Metric | Score |", "|---|---:|"])
    for metric in CHAIN_METRICS:
        lines.append(f"| {metric} | {retrieval_metrics.get(metric, 0.0):.1%} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare plain SFT and Retrieval-Aware SFT with fixed RAG evidence.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--baseline-model-path", default=DEFAULT_BASELINE_SFT_MODEL)
    parser.add_argument("--retrieval-aware-model-path", default=DEFAULT_RETRIEVAL_SFT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_CONFIG.embedding_model)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--max-context-chars", type=int, default=2400)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = MediGuideConfig.from_dict(DEFAULT_CONFIG.to_dict())
    config.embedding_model = args.embedding_model
    cases = load_jsonl(args.questions, args.limit)
    data_module, retrieval_module, gate = build_runtime(config)
    evidence_items = precompute_evidence(cases, args, data_module, retrieval_module, gate)

    baseline_rows = evaluate_model("plain_sft_rag", args.baseline_model_path, evidence_items, args)
    retrieval_rows = evaluate_model("retrieval_aware_sft", args.retrieval_aware_model_path, evidence_items, args)

    baseline_metrics = aggregate(baseline_rows)
    retrieval_metrics = aggregate(retrieval_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plain_sft_rag_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in baseline_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "retrieval_aware_sft_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in retrieval_rows) + "\n",
        encoding="utf-8",
    )
    metrics = {"plain_sft_rag": baseline_metrics, "retrieval_aware_sft": retrieval_metrics}
    (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(baseline_metrics, retrieval_metrics, output_dir / "comparison_report.md")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

