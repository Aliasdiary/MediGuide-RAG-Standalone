"""Run the exported MediGuide SFT model with Transformers.

This script uses the exported MediGuide SFT model with manual Qwen ChatML
prompting and conservative beam sampling. It can run with only a user question,
or with retrieved RAG evidence passed through `--context`.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/MediGuide-RAG-Standalone/"
    "finetune/export/qwen25-3b-mediguide-sft"
)

# Keep the prompt explicit because the exported tokenizer has no chat template.
# The design follows evidence-grounded generation, private self-checking, and
# principle-based safety constraints used in recent RAG/safety work.
SAFETY_SYSTEM_PROMPT = (
    "你是医疗健康科普助手，只提供健康科普信息，不进行诊断、处方或个体化剂量建议。"
    "回答前先在内部完成三项检查：是否存在危险信号，是否涉及用药或剂量，是否有足够证据；不要输出检查过程。"
    "如果用户提供了 RAG 检索证据，只能基于证据回答；证据不足时要直接说明知识依据不足，不要补编来源。"
    "最终回答用 1 到 2 个自然段：先给最重要的安全建议，再简要说明原因和下一步行动。"
    "不要使用固定小标题，不要重复同一句话，不要编造机构、来源、版权声明或图片信息。"
    "最后只保留一句安全声明：本回答仅用于健康科普，不能替代医生诊断或处方。"
)


def build_user_content(question: str, context: str | None) -> str:
    if context:
        return (
            "RAG retrieved evidence:\n"
            f"{context.strip()}\n\n"
            "User question:\n"
            f"{question.strip()}\n\n"
            "Answer in Chinese using only the retrieved evidence. "
            "If the evidence is insufficient, say the evidence is insufficient."
        )
    return question


def build_qwen_chatml_prompt(question: str, system_prompt: str, context: str | None = None) -> str:
    user_content = build_user_content(question, context)
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with exported MediGuide SFT model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default=None, help="Optional RAG retrieved evidence.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=8)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    prompt = build_qwen_chatml_prompt(args.question, SAFETY_SYSTEM_PROMPT, args.context)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=not args.no_sample,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None if args.no_sample else args.temperature,
            top_p=None if args.no_sample else args.top_p,
            early_stopping=True,
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(answer)


if __name__ == "__main__":
    main()
