"""Run MediGuide-SFT with MedQuAD RAG evidence.

This script connects the existing retriever and the exported SFT generator:

Chinese question -> query expansion -> BGE-M3/FAISS + BM25 + RRF ->
parent QA evidence -> Qwen2.5-3B MediGuide-SFT grounded answer.

The design follows a lightweight RAFT-style open-book setting: the generator
receives retrieved documents, should use relevant evidence, and should ignore
distractor or weakly related evidence.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

import torch
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, MediGuideConfig
from finetune.infer_sft import DEFAULT_MODEL_PATH, SAFETY_SYSTEM_PROMPT, build_qwen_chatml_prompt
from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule


FINAL_DISCLAIMER = "本回答仅用于健康科普，不能替代医生诊断或处方。"

QUERY_TERM_MAP = {
    "高血压": "hypertension high blood pressure",
    "血压": "blood pressure",
    "降压药": "antihypertensive medication blood pressure medicine",
    "剂量": "dose dosage",
    "药量": "dose dosage",
    "加倍": "increase dose double dose",
    "停药": "stop medication discontinue medicine",
    "副作用": "side effects adverse effects",
    "抗生素": "antibiotics",
    "过敏": "allergy allergic reaction",
    "胸痛": "chest pain",
    "呼吸困难": "difficulty breathing shortness of breath",
    "发热": "fever",
    "头痛": "headache",
    "咳嗽": "cough",
    "狗咬": "dog bite animal bite rabies tetanus wound care",
    "动物咬": "animal bite rabies tetanus wound care",
    "疫苗": "vaccine vaccination",
    "检查": "medical test examination",
    "治疗": "treatment therapy management",
    "手术": "surgery",
}


def expand_retrieval_query(question: str) -> str:
    """Build a concise bilingual retrieval query without calling an external LLM."""

    additions = [english for chinese, english in QUERY_TERM_MAP.items() if chinese in question]
    if additions:
        return " ".join([question, *additions])
    return question


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


def retrieve_parent_docs(
    query: str,
    data_module: DataPreparationModule,
    retrieval_module: RetrievalOptimizationModule,
    top_k: int,
) -> List[Document]:
    chunks = retrieval_module.hybrid_search(query, top_k=max(top_k * 2, 8))
    parent_docs = data_module.get_parent_documents(chunks)
    return parent_docs[:top_k]


def compact_medquad_content(content: str, answer_chars: int = 900) -> str:
    if "## Answer" not in content:
        return re.sub(r"\s+", " ", content).strip()[:answer_chars]

    question_part, answer_part = content.split("## Answer", 1)
    question_part = question_part.replace("#", "").replace("Question", "").strip()
    answer_part = re.sub(r"\s+", " ", answer_part).strip()
    excerpt = answer_part[:answer_chars].rsplit(" ", 1)[0].strip()
    return f"Evidence question: {question_part}\nEvidence answer excerpt: {excerpt}"


def format_evidence(docs: List[Document], max_chars: int) -> str:
    if not docs:
        return "No relevant MedQuAD evidence was retrieved."

    parts = []
    seen = set()
    total = 0
    evidence_index = 1
    for doc in docs:
        meta = doc.metadata
        key = (meta.get("focus", ""), meta.get("question_type", ""), meta.get("source_url", ""))
        if key in seen:
            continue
        seen.add(key)
        item = (
            f"[资料{evidence_index}]\n"
            f"Title: {meta.get('focus', 'Unknown')}\n"
            f"Organization: {meta.get('source_org', 'Unknown')}\n"
            f"Question type: {meta.get('question_type', 'unknown')}\n"
            f"URL: {meta.get('source_url', '')}\n"
            f"{compact_medquad_content(doc.page_content)}\n"
        )
        if total + len(item) > max_chars:
            break
        parts.append(item)
        total += len(item)
        evidence_index += 1
    return "\n" + ("\n" + "-" * 60 + "\n").join(parts)


def build_grounded_user_prompt(question: str, evidence: str) -> str:
    return f"""你将获得一个用户医疗健康问题和若干条 MedQuAD 检索证据。

请按以下通用标准回答，不要输出思考过程：
1. 先判断用户真正的问题，并在第一句话直接回答。
2. 只使用与问题直接相关的证据；如果证据里有无关资料或长列表，只提取必要信息，不要照搬原文。
3. 如果证据不足以支持结论，明确说明当前检索证据不足，不要用模型常识补充医学结论。
4. 涉及诊断、处方、药物剂量或治疗方案时，只能提供健康科普和就医建议，不能给个体化诊断、处方或剂量。
5. 涉及急性危险信号时，应建议及时就医或急诊，但不能做确定诊断。
6. 使用简体中文，输出 3 到 5 句自然段；不要使用“结论/原因/建议”固定标题，不要重复句子。
7. 如引用证据，可用 [资料1]、[资料2] 这种形式简要标注。

用户问题：
{question}

MedQuAD 检索证据：
{evidence}

请直接输出最终中文回答："""


def normalize_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    text = re.sub(r"(?i)final answer(?: in chinese)?:", "", text).strip()
    text = re.sub(r"(?i)this answer is for health education only[^.\n]*\.", "", text).strip()
    text = text.translate(
        str.maketrans(
            {
                "醫": "医",
                "診": "诊",
                "處": "处",
                "療": "疗",
                "壓": "压",
                "體": "体",
                "藥": "药",
                "議": "议",
                "調": "调",
                "劑": "剂",
            }
        )
    )
    text = dedupe_sentences(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = text.replace(FINAL_DISCLAIMER, "").strip()
    if "不能替代医生" not in text:
        text = f"{text}\n\n{FINAL_DISCLAIMER}".strip()
    return text


def dedupe_sentences(text: str) -> str:
    pieces = re.split(r"(?<=[。！？!?])", text)
    seen = set()
    kept = []
    for piece in pieces:
        sentence = piece.strip()
        if not sentence:
            continue
        normalized = re.sub(r"\s+", "", sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(sentence)
    return "".join(kept) if kept else text


def print_hits(docs: Iterable[Document]) -> None:
    print("\nRAG retrieval hits:")
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
    parser.add_argument("--retrieval-query", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--index-save-path", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=4500)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_values = DEFAULT_CONFIG.to_dict()
    if args.embedding_model:
        config_values["embedding_model"] = args.embedding_model
    if args.index_save_path:
        config_values["index_save_path"] = args.index_save_path
    rag_config = MediGuideConfig.from_dict(config_values)

    data_module, retrieval_module = load_rag_components(rag_config)
    retrieval_query = args.retrieval_query or expand_retrieval_query(args.question)
    print(f"Retrieval query: {retrieval_query}")
    docs = retrieve_parent_docs(retrieval_query, data_module, retrieval_module, args.top_k)
    print_hits(docs)

    evidence = format_evidence(docs, args.max_context_chars)
    grounded_prompt = build_grounded_user_prompt(args.question, evidence)

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

    prompt = build_qwen_chatml_prompt(grounded_prompt, SAFETY_SYSTEM_PROMPT)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print("\nRAG-grounded SFT answer:")
    print(normalize_answer(answer))


if __name__ == "__main__":
    main()
