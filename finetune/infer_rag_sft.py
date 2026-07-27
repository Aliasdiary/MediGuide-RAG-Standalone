"""Run Retrieval-Aware MediGuide-SFT with MedQuAD evidence."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, MediGuideConfig  # noqa: E402
from rag_modules import (  # noqa: E402
    DataPreparationModule,
    EvidenceGate,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)


MEDICAL_QUERY_EXPANSIONS = {
    "dog_bite": {
        "zh": ("狗咬", "动物咬", "被咬", "狂犬"),
        "en": "dog bite animal bite human wound care rabies exposure post exposure prophylaxis tetanus medical care wash wound",
    },
    "dosage": {
        "zh": ("剂量", "加倍", "加量", "多吃", "减量"),
        "en": "medication dose dosage adjustment increase dose double dose prescription safety clinician pharmacist",
    },
    "blood_pressure": {
        "zh": ("降压", "高血压", "血压"),
        "en": "high blood pressure hypertension antihypertensive medication blood pressure treatment",
    },
    "emergency": {
        "zh": ("胸痛", "呼吸困难", "意识", "昏迷", "大出血"),
        "en": "emergency warning signs chest pain shortness of breath loss of consciousness urgent care",
    },
}


SYSTEM_PROMPT = """你是医疗健康科普问答 Agent 的生成模块。
你的输入包含用户问题、RAG 检索证据和证据状态。请遵守：
0. 最终回答必须使用简体中文；英文证据只能作为参考，不能直接输出英文原文。
1. 先回答用户真正问的问题，不要机械翻译或整段复述证据。
2. 只使用与用户主体、疾病/药物/检查项目、问题类型一致的证据。
3. 如果证据不足、主体错配或证据冲突，要明确说明当前证据不足以支持具体回答，并只给通用安全建议。
4. 不做疾病诊断，不提供处方，不给个体化剂量，不要求用户自行调整药物。
5. 对可能需要及时处理的问题，优先给出就医或咨询专业人员的行动建议。
6. 用简洁中文自然段回答，避免重复免责声明、来源版权声明、英文标签和无关机构信息。"""

CHINESE_REWRITE_PROMPT = """你是医疗健康科普问答 Agent 的中文改写模块。
请把候选回答改写为简体中文自然段，并遵守：
1. 只保留与用户问题直接相关的医学科普信息。
2. 不新增诊断、处方或个体化剂量建议。
3. 不输出英文原文、来源版权声明、机构宣传语或重复免责声明。
4. 如果候选回答没有回答用户问题，请基于其有效信息给出简短中文安全回答。"""


def rewrite_retrieval_query(question: str) -> str:
    """Lightweight bilingual query expansion for MedQuAD retrieval."""
    additions: List[str] = []
    for spec in MEDICAL_QUERY_EXPANSIONS.values():
        if any(term in question for term in spec["zh"]):
            additions.append(spec["en"])
    if not additions:
        additions.append("medical health education symptoms treatment medication disease test")
    return f"{question} {' '.join(additions)}"


def load_rag_components(config: MediGuideConfig) -> tuple[DataPreparationModule, RetrievalOptimizationModule]:
    data_module = DataPreparationModule(config.data_path)
    documents = data_module.load_documents()
    chunks = data_module.chunk_documents()
    index_manifest = {
        "dataset_fingerprint": data_module.dataset_fingerprint(),
        "embedding_model": config.embedding_model,
        "dataset_limit": config.dataset_limit,
        "dataset_seed": config.dataset_seed,
        "chunk_strategy": "question-child/full-qa-parent-v1",
    }
    index_module = IndexConstructionModule(
        model_name=config.embedding_model,
        index_save_path=config.index_save_path,
        expected_manifest=index_manifest,
    )
    vectorstore = index_module.load_index()
    if vectorstore is None:
        vectorstore = index_module.build_vector_index(chunks)
        index_module.save_index()
    retrieval_module = RetrievalOptimizationModule(
        vectorstore,
        chunks,
        use_cross_encoder_reranker=config.use_cross_encoder_reranker,
        reranker_model=config.reranker_model,
    )
    return data_module, retrieval_module


def format_evidence(docs: List[Document], max_context_chars: int) -> str:
    if not docs:
        return "当前未检索到能够直接回答该问题的医学证据。"

    blocks = []
    remaining = max_context_chars
    for idx, doc in enumerate(docs, start=1):
        answer = doc.page_content
        if "## Answer" in answer:
            answer = answer.split("## Answer", 1)[1].strip()
        focus = str(doc.metadata.get("focus", "medical evidence")).strip()
        block = f"[证据{idx}] 主题：{focus}\n{answer}"
        if len(block) > remaining:
            block = block[: max(0, remaining)].rstrip()
        if block:
            blocks.append(block)
            remaining -= len(block)
        if remaining <= 0:
            break
    return "\n\n".join(blocks)


def build_user_prompt(question: str, evidence_status: str, evidence_text: str) -> str:
    return f"""用户问题：
{question}

证据状态：
{evidence_status}

RAG 检索证据：
{evidence_text}

请基于上述证据状态和可用证据，用中文回答用户问题。"""


def build_rewrite_prompt(question: str, draft_answer: str) -> str:
    return f"""用户问题：
{question}

候选回答：
{draft_answer}

请输出最终简体中文回答。"""


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalize_answer(answer: str) -> str:
    answer = answer.replace("<|im_end|>", "").strip()
    answer = re.sub(r"(?i)final answer in chinese:\s*", "", answer)
    answer = re.sub(r"(?i)this answer is for health education only.*", "", answer).strip()
    answer = re.sub(r"本回答(仅|只)?用于健康科普.*?(。|$)", "", answer).strip()
    answer = re.sub(r"(不能替代医生诊断或治疗建议。)(\s*\1)+", r"\1", answer)
    return answer.strip()


def mostly_english(text: str) -> bool:
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    return ascii_letters > 40 and ascii_letters > chinese_chars


def run_generation(model, tokenizer, device: str, prompt: str, generation_kwargs: dict) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def generate_answer(args, evidence_status: str, evidence_text: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, build_user_prompt(args.question, evidence_status, evidence_text))

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
            }
        )
    else:
        generation_kwargs.update({"do_sample": False, "num_beams": 1})

    answer = normalize_answer(run_generation(model, tokenizer, device, prompt, generation_kwargs))
    if args.force_chinese and mostly_english(answer):
        rewrite_prompt = build_chat_prompt(tokenizer, CHINESE_REWRITE_PROMPT, build_rewrite_prompt(args.question, answer))
        answer = normalize_answer(run_generation(model, tokenizer, device, rewrite_prompt, generation_kwargs))
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG-grounded MediGuide-SFT inference.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_CONFIG.rag_sft_model_path))
    parser.add_argument("--embedding-model", default=DEFAULT_CONFIG.embedding_model)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--max-context-chars", type=int, default=2400)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--no-force-chinese", dest="force_chinese", action="store_false")
    parser.set_defaults(force_chinese=True)
    args = parser.parse_args()

    rag_config = MediGuideConfig.from_dict(DEFAULT_CONFIG.to_dict())
    rag_config.embedding_model = args.embedding_model
    rag_config.use_cross_encoder_reranker = args.use_reranker

    data_module, retrieval_module = load_rag_components(rag_config)
    gate = EvidenceGate(
        min_support=rag_config.evidence_gate_min_score,
        max_docs=rag_config.evidence_gate_max_docs,
    )

    retrieval_query = rewrite_retrieval_query(args.question)
    retrieval_top_k = gate.dynamic_top_k(args.question, args.top_k)
    child_hits = retrieval_module.hybrid_search(retrieval_query, top_k=retrieval_top_k)
    parent_hits = data_module.get_parent_documents(child_hits)
    gate_result = gate.assess(args.question, parent_hits, route=gate.infer_route(args.question))

    print(f"Retrieval query: {retrieval_query}")
    print(f"Evidence status: {gate_result.status}")
    print("\nRAG retrieval hits:")
    for idx, doc in enumerate(parent_hits, start=1):
        print(
            f"{idx}. {doc.metadata.get('focus')} | {doc.metadata.get('source_org')}/"
            f"{doc.metadata.get('question_type')} | support={doc.metadata.get('evidence_support', '-')}"
        )
        print(f"   status={doc.metadata.get('evidence_status', '-')} reason={doc.metadata.get('evidence_gate_reason', '-')}")
        print(f"   {doc.metadata.get('source_url')}")

    evidence_text = format_evidence(gate_result.usable_docs, args.max_context_chars)
    answer = generate_answer(args, gate_result.status, evidence_text)
    print("\nRAG-grounded SFT answer:")
    print(answer)


if __name__ == "__main__":
    main()
