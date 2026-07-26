"""Run the exported MediGuide SFT model with Transformers.

This is the stable fallback path when vLLM is unavailable in a constrained
AutoDL environment. It expects the LoRA adapter to have been merged/exported
with `finetune/export_qwen25_3b_mediguide.yaml`.
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

# Kept as unicode escapes so the prompt survives Windows/AutoDL encoding hops.
SAFETY_SYSTEM_PROMPT = (
    "\u4f60\u662f\u533b\u7597\u5065\u5eb7\u79d1\u666e\u52a9\u624b\u3002"
    "\u53ea\u56de\u7b54\u4e00\u6bb5\u7b80\u77ed\u4e2d\u6587\uff0c"
    "\u4e0d\u8981\u5217\u865a\u6784\u6765\u6e90\u3001\u673a\u6784\u3001"
    "\u7248\u6743\u58f0\u660e\u6216\u56fe\u7247\u4fe1\u606f\u3002"
    "\u4e0d\u8fdb\u884c\u8bca\u65ad\uff0c\u4e0d\u5f00\u5904\u65b9\uff0c"
    "\u4e0d\u63d0\u4f9b\u4e2a\u4f53\u5316\u5242\u91cf\u3002"
    "\u7528\u836f\u95ee\u9898\u8981\u63d0\u793a\u4e0d\u8981\u81ea\u884c"
    "\u8c03\u6574\u5242\u91cf\uff0c\u5efa\u8bae\u54a8\u8be2\u533b\u751f"
    "\u6216\u836f\u5e08\u3002"
    "\u80f8\u75db\u3001\u547c\u5438\u56f0\u96be\u3001\u610f\u8bc6\u969c\u788d\u3001"
    "\u4e25\u91cd\u8fc7\u654f\u7b49\u5371\u9669\u4fe1\u53f7\u8981\u4f18\u5148"
    "\u5efa\u8bae\u53ca\u65f6\u5c31\u533b\u6216\u6025\u8bca\u3002"
    "\u6700\u540e\u53ea\u4fdd\u7559\u4e00\u6b21\u201c\u672c\u56de\u7b54"
    "\u4ec5\u7528\u4e8e\u5065\u5eb7\u79d1\u666e\uff0c\u4e0d\u80fd\u66ff\u4ee3"
    "\u533b\u751f\u8bca\u65ad\u6216\u5904\u65b9\u3002\u201d"
)


def build_qwen_chatml_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def qwen_eos_token_ids(tokenizer: AutoTokenizer) -> list[int]:
    token_ids = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.append(token_id)
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    return sorted(set(token_ids))


def clean_answer(answer: str) -> str:
    answer = answer.split("<|im_end|>", 1)[0].strip()
    hallucinated_markers = [
        "\u4e01\u9999\u533b\u751f",
        "\u672c\u56de\u7b54\u5185\u5bb9\u672a\u7ecf\u6388\u6743",
        "\u4efb\u4f55\u56fe\u7247",
        "\u5546\u4e1a\u76ee\u7684",
        "\u5546\u4e1a\u7528\u9014",
    ]
    for marker in hallucinated_markers:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    sentences = re.split(r"(?<=[。！？])", answer)
    kept = []
    disclaimer_seen = False
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        is_disclaimer = "\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f\u8bca\u65ad" in sentence
        if is_disclaimer and disclaimer_seen:
            continue
        disclaimer_seen = disclaimer_seen or is_disclaimer
        kept.append(sentence)
        if len("".join(kept)) >= 180:
            break
    return "".join(kept).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with exported MediGuide SFT model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    prompt = build_qwen_chatml_prompt(args.question, SAFETY_SYSTEM_PROMPT)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.25,
            no_repeat_ngram_size=6,
            eos_token_id=qwen_eos_token_ids(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(clean_answer(answer))


if __name__ == "__main__":
    main()
