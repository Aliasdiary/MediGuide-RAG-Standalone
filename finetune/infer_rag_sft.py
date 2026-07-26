"""Run MediGuide-SFT with retrieved MedQuAD evidence.

This is the bridge between the RAG system and the SFT model:

Chinese question -> MedQuAD hybrid retrieval -> parent QA evidence ->
Qwen2.5-3B MediGuide-SFT answer.

It intentionally keeps `infer_sft.py` unchanged, because that file is the
validated standalone SFT inference path.
"""

from __future__ import annotations

import argparse
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
    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    return data_module, retrieval_module


def retrieve_parent_docs(
    question: str,
    data_module: DataPreparationModule,
    retrieval_module: RetrievalOptimizationModule,
    top_k: int,
) -> List[Document]:
    chunks = retrieval_module.hybrid_search(question, top_k=top_k)
    return data_module.get_parent_documents(chunks)


def format_evidence(docs: List[Document], max_chars: int) -> str:
    if not docs:
        return "No relevant MedQuAD evidence was retrieved."

    parts = []
    current = 0
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        item = (
            f"[Evidence {index}]\n"
            f"Title: {meta.get('focus', 'Unknown')}\n"
            f"Organization: {meta.get('source_org', 'Unknown')}\n"
            f"Question type: {meta.get('question_type', 'unknown')}\n"
            f"URL: {meta.get('source_url', '')}\n"
            f"License: {meta.get('license', 'CC BY 4.0')}\n"
            f"{doc.page_content}\n"
        )
        if current + len(item) > max_chars:
            break
        parts.append(item)
        current += len(item)
    return "\n" + ("-" * 60 + "\n").join(parts)


def build_rag_grounded_question(question: str, evidence: str) -> str:
    return (
        "请基于下面的 MedQuAD 检索资料回答用户问题。"
        "如果资料不足，请明确说明当前检索证据不足，不要编造来源或结论。"
        "回答要使用中文，不能进行诊断、处方或个体化剂量建议；"
        "涉及用药调整时建议咨询医生或药师。\n\n"
        f"用户问题：{question}\n\n"
        f"MedQuAD 检索资料：\n{evidence}\n\n"
        "请给出简洁、可读、带安全边界的回答。"
    )


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
    docs = retrieve_parent_docs(
        args.retrieval_query or args.question,
        data_module,
        retrieval_module,
        top_k=args.top_k,
    )
    print_hits(docs)

    evidence = format_evidence(docs, max_chars=args.max_context_chars)
    grounded_question = build_rag_grounded_question(args.question, evidence)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
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
    print(answer)


if __name__ == "__main__":
    main()
