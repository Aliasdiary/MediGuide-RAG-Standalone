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


def compact_medquad_content(content: str, answer_chars: int = 700) -> str:
    if "## Answer" not in content:
        return content.strip()[:answer_chars]

    question_part, answer_part = content.split("## Answer", 1)
    question_part = question_part.replace("## Question", "").strip()
    answer_part = re.sub(r"\s+", " ", answer_part).strip()
    answer_excerpt = answer_part[:answer_chars].rsplit(" ", 1)[0].strip()
    return (
        f"Evidence question: {question_part}\n"
        f"Evidence answer excerpt: {answer_excerpt}"
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
    seen = set()
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        dedupe_key = (
            meta.get("focus", ""),
            meta.get("question_type", ""),
            meta.get("source_url", ""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        content = compact_medquad_content(doc.page_content)
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
        "\u4f60\u662f\u533b\u7597\u5065\u5eb7\u79d1\u666e\u52a9\u624b\uff0c\u4f60\u7684\u4efb\u52a1\u662f"
        "\u57fa\u4e8e MedQuAD \u68c0\u7d22\u8bc1\u636e\u76f4\u63a5\u56de\u7b54\u7528\u6237\u95ee\u9898\uff0c"
        "\u800c\u4e0d\u662f\u7ffb\u8bd1\u6216\u590d\u8ff0\u68c0\u7d22\u8d44\u6599\u3002\n\n"
        "\u901a\u7528\u751f\u6210\u6807\u51c6\uff1a\n"
        "1. \u5148\u5224\u65ad\u7528\u6237\u771f\u6b63\u5728\u95ee\u4ec0\u4e48\uff0c\u7b2c\u4e00\u53e5\u5fc5\u987b\u76f4\u63a5\u56de\u7b54\u8fd9\u4e2a\u95ee\u9898\u3002\n"
        "2. \u53ea\u4f7f\u7528\u68c0\u7d22\u8bc1\u636e\u4e2d\u4e0e\u95ee\u9898\u76f4\u63a5\u76f8\u5173\u7684\u4fe1\u606f\uff0c"
        "\u5ffd\u7565\u65e0\u5173\u80cc\u666f\u3001\u957f\u5217\u8868\u3001\u91cd\u590d\u8bc1\u636e\u548c\u8dd1\u9898\u5185\u5bb9\u3002\n"
        "3. \u4e0d\u8981\u7167\u642c\u82f1\u6587\u8bc1\u636e\uff0c\u5fc5\u987b\u7528\u7b80\u4f53\u4e2d\u6587\u5f52\u7eb3\u3001\u538b\u7f29\u548c\u8f6c\u8ff0\u3002\n"
        "4. \u5982\u679c\u8bc1\u636e\u4e0d\u8db3\u4ee5\u652f\u6301\u7ed3\u8bba\uff0c\u8bf4\u660e\u201c\u5f53\u524d\u68c0\u7d22\u8bc1\u636e\u4e0d\u8db3\u201d\uff0c"
        "\u4e0d\u8981\u7f16\u9020\u8bca\u65ad\u3001\u5904\u65b9\u3001\u5242\u91cf\u6216\u7597\u6548\u3002\n"
        "5. \u6d89\u53ca\u7528\u836f\u3001\u5242\u91cf\u3001\u6cbb\u7597\u65b9\u6848\u65f6\uff0c\u53ea\u80fd\u7ed9\u5065\u5eb7\u79d1\u666e\u548c\u5c31\u533b\u5efa\u8bae\uff0c"
        "\u4e0d\u80fd\u7ed9\u4e2a\u4f53\u5316\u5242\u91cf\u6216\u8981\u6c42\u7528\u6237\u81ea\u884c\u6539\u836f\u3002\n"
        "6. \u6d89\u53ca\u6025\u6027\u5371\u9669\u4fe1\u53f7\u65f6\uff0c\u5e94\u63d0\u9192\u53ca\u65f6\u5c31\u533b\u6216\u6025\u8bca\uff0c\u4f46\u4e0d\u505a\u786e\u5b9a\u8bca\u65ad\u3002\n"
        "7. \u8f93\u51fa 3 \u5230 5 \u53e5\u81ea\u7136\u6bb5\uff0c\u4e0d\u8981\u7528\u201c\u7ed3\u8bba/\u539f\u56e0/\u5efa\u8bae\u201d\u8fd9\u79cd\u56fa\u5b9a\u6807\u9898\uff0c"
        "\u4e0d\u8981\u8f93\u51fa\u82f1\u6587\u63d0\u793a\u8bed\uff0c\u4e0d\u8981\u91cd\u590d\u53e5\u5b50\u3002\n\n"
        f"\u7528\u6237\u95ee\u9898\uff1a{question}\n\n"
        f"MedQuAD \u68c0\u7d22\u8bc1\u636e\uff1a\n{evidence}\n\n"
        "\u8bf7\u73b0\u5728\u76f4\u63a5\u8f93\u51fa\u7b80\u4f53\u4e2d\u6587\u56de\u7b54\uff0c"
        "\u53ea\u8f93\u51fa\u6700\u7ec8\u7b54\u6848\u672c\u8eab\uff1a"
    )


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    cleanup_patterns = [
        r"(?i)final answer(?: in chinese)?:",
        r"(?i)this answer is for health education only[^.\n]*\.",
        r"\u8bf7\u73b0\u5728\u76f4\u63a5\u8f93\u51fa\u7b80\u4f53\u4e2d\u6587\u56de\u7b54\uff0c?",
    ]
    for pattern in cleanup_patterns:
        answer = re.sub(pattern, "", answer).strip()
    traditional_to_simplified = {
        "\u91ab": "\u533b",
        "\u8a3a": "\u8bca",
        "\u8655": "\u5904",
        "\u7642": "\u7597",
        "\u58d3": "\u538b",
        "\u9ad4": "\u4f53",
        "\u85e5": "\u836f",
        "\u8b70": "\u8bae",
        "\u8f49": "\u8f6c",
        "\u8abf": "\u8c03",
        "\u91cf": "\u91cf",
    }
    answer = answer.translate(str.maketrans(traditional_to_simplified))
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    if FINAL_DISCLAIMER in answer:
        answer = answer.replace(FINAL_DISCLAIMER, "").strip()
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
    print("\nRAG-grounded SFT answer:")
    print(clean_answer(answer))


if __name__ == "__main__":
    main()
