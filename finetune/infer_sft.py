"""Run the exported MediGuide SFT model with Transformers.

This script uses the exported MediGuide SFT model with manual Qwen ChatML
prompting and conservative beam sampling. It can run with only a user question,
or with retrieved RAG evidence passed through `--context`.
"""

from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/MediGuide-RAG-Standalone/"
    "finetune/export/qwen25-3b-mediguide-sft"
)

FINAL_DISCLAIMER = (
    "\u672c\u56de\u7b54\u4ec5\u7528\u4e8e\u5065\u5eb7\u79d1\u666e\uff0c"
    "\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f\u8bca\u65ad\u6216\u5904\u65b9\u3002"
)

# ASCII-only prompt text avoids Windows/AutoDL encoding corruption. The model is
# still instructed to answer in Chinese, and the final disclaimer renders in
# Chinese through Unicode escapes.
SAFETY_SYSTEM_PROMPT = (
    "You are a medical health education assistant. Answer in Chinese. "
    "You must answer the user's actual question first. Do not start with a disclaimer "
    "and do not output only a disclaimer. "
    "Before answering, privately check emergency signals, medication/dosage risk, "
    "and evidence sufficiency; do not reveal this private check. "
    "If RAG evidence is provided, ground the answer only on that evidence. "
    "If evidence is insufficient, say the evidence is insufficient instead of inventing sources. "
    "If no RAG evidence is provided, give general non-personalized health education. "
    "Use one or two concise natural paragraphs. First give the most important safety action, "
    "then briefly explain why and what to do next. "
    "Do not use section headings. Do not repeat the same sentence. "
    "Do not fabricate organizations, URLs, copyrights, images, or source claims. "
    f"The final sentence must appear exactly once and only at the end: {FINAL_DISCLAIMER}"
)

SAFETY_HINTS = [
    (
        ("\u72d7\u54ac", "\u72ac\u54ac", "\u52a8\u7269\u54ac", "\u732b\u54ac", "\u54ac\u4f24"),
        "\u98ce\u9669\u63d0\u793a\uff1a\u52a8\u7269\u54ac\u4f24\u540e\uff0c"
        "\u5148\u7528\u6d41\u52a8\u6e05\u6c34\u548c\u80a5\u7682\u51b2\u6d17\u4f24\u53e3\uff0c"
        "\u5e76\u5c3d\u5feb\u5230\u533b\u9662\u6216\u72ac\u4f24\u95e8\u8bca\u8bc4\u4f30"
        "\u72c2\u72ac\u75c5\u66b4\u9732\u548c\u7834\u4f24\u98ce\u9884\u9632\u3002",
    ),
    (
        ("\u80f8\u75db", "\u547c\u5438\u56f0\u96be", "\u610f\u8bc6\u4e0d\u6e05", "\u660f\u8ff7"),
        "Safety hint: for chest pain, breathing difficulty, or altered consciousness, "
        "prioritize urgent/emergency care.",
    ),
    (
        ("\u5242\u91cf", "\u52a0\u500d", "\u964d\u538b\u836f", "\u6297\u751f\u7d20"),
        "Safety hint: for medication dosage questions, advise not to self-adjust medication "
        "and to consult a clinician or pharmacist.",
    ),
]


def build_safety_hint(question: str) -> str | None:
    for keywords, hint in SAFETY_HINTS:
        if any(keyword in question for keyword in keywords):
            return hint
    return None


def build_user_content(question: str, context: str | None) -> str:
    hint = build_safety_hint(question)
    risk_hint = f"Risk hint: {hint}\n" if hint else ""
    if context:
        return (
            "RAG retrieved evidence:\n"
            f"{context.strip()}\n\n"
            "User question:\n"
            f"{question.strip()}\n\n"
            f"{risk_hint}"
            "Answer in Chinese using only the retrieved evidence. "
            "If the evidence is insufficient, say the evidence is insufficient."
        )
    return (
        f"User question:\n{question.strip()}\n\n"
        f"{risk_hint}"
        "Answer the question directly in Chinese. Do not begin with the safety disclaimer."
    )


def build_qwen_chatml_prompt(question: str, system_prompt: str, context: str | None = None) -> str:
    user_content = build_user_content(question, context)
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return answer

    blocked_terms = [
        "\u4e01\u9999\u533b\u751f",
        "\u56fe\u7247",
        "\u7248\u6743",
        "\u5546\u4e1a\u7528\u9014",
        "\u672a\u7ecf\u6388\u6743",
    ]
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[\u3002\uff01\uff1f!?])\s*", answer)
        if sentence.strip()
    ]
    cleaned = []
    seen = set()
    for sentence in sentences:
        if any(term in sentence for term in blocked_terms):
            continue
        sentence = sentence.replace(FINAL_DISCLAIMER, "").strip()
        if not sentence or sentence in seen:
            continue
        cleaned.append(sentence)
        seen.add(sentence)

    body = " ".join(cleaned).strip()
    body = re.sub(
        r"(\u672c\u56de\u7b54|\u8be5\u56de\u7b54).{0,50}?\u4e0d\u80fd\u66ff\u4ee3.{0,40}?\u3002",
        "",
        body,
    ).strip()
    return f"{body} {FINAL_DISCLAIMER}".strip() if body else FINAL_DISCLAIMER


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
    print(clean_answer(answer))


if __name__ == "__main__":
    main()
