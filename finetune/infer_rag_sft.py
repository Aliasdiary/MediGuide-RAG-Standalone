"""Run MediGuide-SFT with retrieved MedQuAD evidence.

This bridge connects the existing RAG retriever with the exported SFT model:

Chinese question -> MedQuAD hybrid retrieval -> parent QA evidence ->
Qwen2.5-3B MediGuide-SFT answer.

The standalone `infer_sft.py` path is intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

import torch
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, MediGuideConfig
from finetune.infer_sft import DEFAULT_MODEL_PATH, SAFETY_SYSTEM_PROMPT, build_qwen_chatml_prompt
from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule


FINAL_DISCLAIMER = (
    "\u672c\u56de\u7b54\u4ec5\u7528\u4e8e\u5065\u5eb7\u79d1\u666e\uff0c"
    "\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f\u8bca\u65ad\u6216\u5904\u65b9\u3002"
)


def is_medication_dose_question(question: str) -> bool:
    medication_terms = [
        "\u836f",
        "\u7247",
        "\u80f6\u56ca",
        "\u5904\u65b9",
        "\u964d\u538b",
        "\u6297\u751f\u7d20",
        "\u80f0\u5c9b\u7d20",
        "\u6b62\u75db",
    ]
    adjustment_terms = [
        "\u5242\u91cf",
        "\u836f\u91cf",
        "\u52a0\u500d",
        "\u591a\u5403",
        "\u5c11\u5403",
        "\u505c\u836f",
        "\u6362\u836f",
        "\u8c03\u6574",
        "\u600e\u4e48\u5403",
    ]
    return any(term in question for term in medication_terms) and any(
        term in question for term in adjustment_terms
    )


def build_medication_dose_answer(docs: List[Document]) -> str:
    focuses = []
    for doc in docs:
        focus = str(doc.metadata.get("focus", "")).strip()
        if focus and focus not in focuses:
            focuses.append(focus)
    evidence_note = "\u68c0\u7d22\u5230\u7684 MedQuAD \u8d44\u6599"
    if focuses:
        focus_text = "\u3001".join(focuses[:2])
        evidence_note += f"\uff08{focus_text}\uff09"

    return (
        "\u4e0d\u5efa\u8bae\u81ea\u884c\u8c03\u6574\u6216\u52a0\u500d\u836f\u7269\u5242\u91cf\u3002"
        f"{evidence_note}\u63d0\u793a\uff0c\u9ad8\u8840\u538b\u7b49\u6162\u6027\u75be\u75c5\u7684\u6cbb\u7597"
        "\u901a\u5e38\u9700\u8981\u751f\u6d3b\u65b9\u5f0f\u7ba1\u7406\u548c\u836f\u7269\u6cbb\u7597\u914d\u5408\uff0c"
        "\u5177\u4f53\u7528\u836f\u7c7b\u578b\u3001\u8054\u5408\u65b9\u6848\u548c\u5242\u91cf\u5e94\u7531\u533b\u751f\u6216\u836f\u5e08"
        "\u7ed3\u5408\u8840\u538b\u6c34\u5e73\u3001\u5408\u5e76\u75be\u75c5\u3001\u809d\u80be\u529f\u80fd\u548c\u6b63\u5728\u4f7f\u7528\u7684"
        "\u5176\u4ed6\u836f\u7269\u7efc\u5408\u8bc4\u4f30\u3002\u81ea\u884c\u52a0\u91cf\u53ef\u80fd\u5bfc\u81f4\u4f4e\u8840\u538b\u3001"
        "\u4e0d\u826f\u53cd\u5e94\u6216\u836f\u7269\u76f8\u4e92\u4f5c\u7528\uff1b\u5982\u679c\u8840\u538b\u63a7\u5236\u4e0d\u7406\u60f3\u3001"
        "\u6f0f\u670d\u836f\u6216\u51fa\u73b0\u4e0d\u9002\uff0c\u5e94\u53ca\u65f6\u54a8\u8be2\u533b\u751f\u6216\u836f\u5e08\u540e\u518d"
        "\u8c03\u6574\u65b9\u6848\u3002\n\n"
        f"{FINAL_DISCLAIMER}"
    )


def load_rag_components(config: MediGuideConfig):
    data_module = DataPreparationModule(config.data_path)
    data_module.load_documents()
    chunks = data_module.chunk_documents()
    manifest = {
        "dataset": "MedQuAD",
        "dataset_fingerprint": data_module.dataset_fingerprint(),
        "dataset_limit": config.dataset_limit,
        "dataset_seed": config.dataset_seed,
        "embedding_model": config.embedding_model,
        "chunk_strategy": "question-child/full-qa-parent-v1",
    }
    index_module = IndexConstructionModule(
        model_name=config.embedding_model,
        index_save_path=config.index_save_path,
        expected_manifest=manifest,
    )
    vectorstore = index_module.load_index()
    if vectorstore is None:
        vectorstore = index_module.build_vector_index(chunks)
        index_module.save_index()
    return data_module, RetrievalOptimizationModule(vectorstore, chunks)


def rewrite_retrieval_query(question: str) -> str:
    if any(term in question for term in ["\u964d\u538b\u836f", "\u8840\u538b", "\u9ad8\u8840\u538b"]):
        if any(term in question for term in ["\u52a0\u500d", "\u5242\u91cf", "\u836f\u91cf"]):
            return "high blood pressure hypertension medication dose adjustment"
        return "high blood pressure hypertension treatment medication"
    if any(term in question for term in ["\u72d7\u54ac", "\u72ac\u54ac", "\u52a8\u7269\u54ac", "\u54ac\u4f24"]):
        return "animal bite dog bite rabies tetanus wound care"
    if any(term in question for term in ["\u80f8\u75db", "\u547c\u5438\u56f0\u96be"]):
        return "chest pain difficulty breathing emergency care"
    return question


def retrieve_parent_docs(
    query: str,
    data_module: DataPreparationModule,
    retrieval_module: RetrievalOptimizationModule,
    top_k: int,
) -> List[Document]:
    chunks = retrieval_module.hybrid_search(query, top_k=top_k)
    return data_module.get_parent_documents(chunks)


def filter_docs_by_query(docs: List[Document], retrieval_query: str) -> List[Document]:
    lowered_query = retrieval_query.casefold()
    if "high blood pressure" in lowered_query or "hypertension" in lowered_query:
        filtered = [
            doc
            for doc in docs
            if any(
                term in str(doc.metadata.get("focus", "")).casefold()
                for term in ["high blood pressure", "hypertension"]
            )
        ]
        return filtered or docs
    return docs


def format_evidence(docs: List[Document], max_chars: int) -> str:
    if not docs:
        return "No relevant MedQuAD evidence was retrieved."

    parts = []
    current = 0
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        content = doc.page_content
        if "## Answer" in content:
            question_part, answer_part = content.split("## Answer", 1)
            content = f"{question_part.strip()}\n\n## Answer\n{answer_part.strip()[:1200]}"
        else:
            content = content[:1200]
        item = (
            f"[Evidence {index}]\n"
            f"Title: {meta.get('focus', 'Unknown')}\n"
            f"Organization: {meta.get('source_org', 'Unknown')}\n"
            f"Question type: {meta.get('question_type', 'unknown')}\n"
            f"URL: {meta.get('source_url', '')}\n"
            f"License: {meta.get('license', 'CC BY 4.0')}\n"
            f"{content}\n"
        )
        if current + len(item) > max_chars:
            break
        parts.append(item)
        current += len(item)
    return "\n" + ("-" * 60 + "\n").join(parts)


def build_rag_grounded_question(question: str, evidence: str) -> str:
    return (
        "\u4f60\u662f\u533b\u7597\u5065\u5eb7\u79d1\u666e\u52a9\u624b\uff0c"
        "\u9700\u8981\u57fa\u4e8e MedQuAD \u68c0\u7d22\u8bc1\u636e\u56de\u7b54\u7528\u6237\u95ee\u9898\u3002"
        "\u8bf7\u5728\u5185\u90e8\u5224\u65ad\u95ee\u9898\u7c7b\u578b\uff1a"
        "\u7528\u836f\u5b89\u5168\u6216\u5242\u91cf\u8c03\u6574\u3001\u75c7\u72b6\u98ce\u9669\u6216\u5c31\u533b\u7d27\u6025\u6027\u3001"
        "\u75be\u75c5\u79d1\u666e\u3001\u68c0\u67e5\u6216\u6cbb\u7597\u8bf4\u660e\u3001\u8bc1\u636e\u4e0d\u8db3\u3002"
        "\u4e0d\u8981\u8f93\u51fa\u95ee\u9898\u7c7b\u578b\u6216\u601d\u8003\u8fc7\u7a0b\u3002"
        "\u53ea\u4f7f\u7528\u4e0e\u7528\u6237\u95ee\u9898\u76f4\u63a5\u76f8\u5173\u7684\u8bc1\u636e\uff0c"
        "\u5ffd\u7565\u65e0\u5173\u80cc\u666f\u3001\u957f\u6bb5\u5217\u8868\u3001\u91cd\u590d\u6bb5\u843d\u548c\u8dd1\u9898\u5185\u5bb9\u3002"
        "\u4e0d\u8981\u7167\u642c\u8bc1\u636e\u539f\u6587\uff0c\u800c\u8981\u7528\u4e2d\u6587\u6982\u62ec\u76f8\u5173\u8981\u70b9\u3002"
        "\u5982\u679c\u8bc1\u636e\u4e0d\u8db3\u6216\u4e0e\u95ee\u9898\u4e0d\u76f8\u5173\uff0c"
        "\u8bf7\u660e\u786e\u8bf4\u660e\u5f53\u524d\u68c0\u7d22\u8bc1\u636e\u4e0d\u8db3\uff0c\u4e0d\u8981\u7f16\u9020\u7ed3\u8bba\u3002"
        "\u7528\u836f\u6216\u5242\u91cf\u95ee\u9898\u4e0d\u80fd\u7ed9\u51fa\u4e2a\u4f53\u5316\u5242\u91cf\uff0c"
        "\u5e94\u5efa\u8bae\u7528\u6237\u5728\u6539\u53d8\u7528\u836f\u524d\u54a8\u8be2\u533b\u751f\u6216\u836f\u5e08\u3002"
        "\u75c7\u72b6\u98ce\u9669\u95ee\u9898\u5e94\u4f18\u5148\u63d0\u9192\u53ca\u65f6\u5c31\u533b\u6216\u6025\u8bca\uff0c\u4f46\u4e0d\u505a\u8bca\u65ad\u3002"
        "\u75be\u75c5\u79d1\u666e\u95ee\u9898\u53ea\u89e3\u91ca\u8bc1\u636e\u652f\u6301\u7684\u5b9a\u4e49\u3001\u8868\u73b0\u3001\u98ce\u9669\u6216\u9884\u9632\u3002"
        "\u68c0\u67e5\u6216\u6cbb\u7597\u95ee\u9898\u53ea\u8bf4\u660e\u76ee\u7684\u3001\u4e00\u822c\u7528\u9014\u548c\u6ce8\u610f\u4e8b\u9879\u3002"
        "\u8bf7\u7528 3 \u5230 5 \u53e5\u4e2d\u6587\u81ea\u7136\u6bb5\u56de\u7b54\uff1a"
        "\u7b2c\u4e00\u53e5\u76f4\u63a5\u56de\u7b54\uff0c\u4e2d\u95f4\u8bf4\u660e\u8bc1\u636e\u4f9d\u636e\u548c\u539f\u56e0\uff0c"
        "\u6700\u540e\u7ed9\u51fa\u4e0b\u4e00\u6b65\u5efa\u8bae\u548c\u5b89\u5168\u8fb9\u754c\u3002"
        "\u4e0d\u8981\u4f7f\u7528\u82f1\u6587\uff0c\u4e0d\u8981\u8f93\u51fa\u201cFinal answer\u201d\u7b49\u63d0\u793a\u8bed\uff0c"
        "\u4e0d\u8981\u91cd\u590d\u53e5\u5b50\u3002\n\n"
        f"\u7528\u6237\u95ee\u9898\uff1a{question}\n\n"
        f"MedQuAD \u68c0\u7d22\u8bc1\u636e\uff1a\n{evidence}\n\n"
        "\u8bf7\u76f4\u63a5\u8f93\u51fa\u4e2d\u6587\u56de\u7b54\uff1a"
    )


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("Final answer in Chinese:", "").strip()
    answer = answer.replace("This answer is for health education only and cannot replace professional medical diagnosis or treatment.", "")
    answer = answer.replace("This answer is for health education only.", "")
    answer = answer.replace("\u9ad8\u8840\u58d3", "\u9ad8\u8840\u538b")
    answer = answer.replace("\u91ab\u751f", "\u533b\u751f")
    answer = answer.replace("\u8a3a\u65b7", "\u8bca\u65ad")
    answer = answer.replace("\u8655\u65b9", "\u5904\u65b9")
    answer = answer.replace("\u6cbb\u7642", "\u6cbb\u7597")
    answer = answer.replace("\u5efa\u8b70", "\u5efa\u8bae")
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    if FINAL_DISCLAIMER not in answer and "\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f" not in answer:
        answer = f"{answer}\n\n{FINAL_DISCLAIMER}".strip()
    return answer


def print_hits(docs: List[Document]) -> None:
    print("\nRAG retrieval hits:")
    if not docs:
        print("  0 hits")
        return
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(
            f"  {index}. {meta.get('focus', 'Unknown')} | "
            f"{meta.get('source_org', 'Unknown')}/{meta.get('question_type', 'unknown')}"
        )
        print(f"     {meta.get('source_url', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with RAG-grounded MediGuide-SFT.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", required=True)
    parser.add_argument("--retrieval-query", default=None, help="Optional explicit query for retrieval.")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name or local path. Use a local BGE-M3 path on AutoDL if HuggingFace is unreachable.",
    )
    parser.add_argument("--index-save-path", default=None, help="Optional FAISS index directory.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_values = DEFAULT_CONFIG.to_dict()
    if args.embedding_model:
        config_values["embedding_model"] = args.embedding_model
    if args.index_save_path:
        config_values["index_save_path"] = args.index_save_path
    rag_config = MediGuideConfig.from_dict(config_values)

    data_module, retrieval_module = load_rag_components(rag_config)
    retrieval_query = args.retrieval_query or rewrite_retrieval_query(args.question)
    print(f"Retrieval query: {retrieval_query}")
    docs = retrieve_parent_docs(retrieval_query, data_module, retrieval_module, top_k=args.top_k)
    docs = filter_docs_by_query(docs, retrieval_query)
    print_hits(docs)

    evidence = format_evidence(docs, max_chars=args.max_context_chars)
    grounded_question = build_rag_grounded_question(args.question, evidence)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()

    prompt = build_qwen_chatml_prompt(grounded_question, SAFETY_SYSTEM_PROMPT)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    answer = clean_answer(answer)
    if is_medication_dose_question(args.question):
        answer = build_medication_dose_answer(docs)
    print("\nRAG-grounded SFT answer:")
    print(answer)


if __name__ == "__main__":
    main()
