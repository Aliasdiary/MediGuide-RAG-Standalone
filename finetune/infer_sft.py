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

# Unicode escapes avoid Windows/AutoDL encoding corruption while keeping the
# runtime prompt identical to the validated Chinese prompt.
SAFETY_SYSTEM_PROMPT = (
    "\u4f60\u662f\u533b\u7597\u5065\u5eb7\u79d1\u666e\u52a9\u624b\uff0c"
    "\u53ea\u63d0\u4f9b\u5065\u5eb7\u79d1\u666e\u4fe1\u606f\uff0c"
    "\u4e0d\u8fdb\u884c\u8bca\u65ad\u3001\u4e0d\u63d0\u4f9b\u5904\u65b9"
    "\u6216\u4e2a\u4f53\u5316\u5242\u91cf\u3002"
    "\u8f93\u51fa\u8981\u5b8c\u6574\u4f46\u514b\u5236\uff0c\u6bcf\u884c\u4e00\u5230\u4e24\u53e5\uff0c"
    "\u4e0d\u8981\u7f16\u9020\u673a\u6784\u3001"
    "\u6765\u6e90\u3001\u7248\u6743\u58f0\u660e\u6216\u56fe\u7247\u4fe1\u606f\u3002"
    "\u4e25\u683c\u6309\u4ee5\u4e0b\u56db\u884c\u683c\u5f0f\u8f93\u51fa\uff1a\n"
    "\u7ed3\u8bba\uff1a\n"
    "\u539f\u56e0\uff1a\n"
    "\u5efa\u8bae\uff1a\n"
    "\u5b89\u5168\u8bf4\u660e\uff1a"
)


def build_user_content(question: str, context: str | None) -> str:
    if context:
        return (
            "RAG retrieved evidence:\n"
            f"{context.strip()}\n\n"
            "User question:\n"
            f"{question.strip()}\n\n"
            "Answer using the retrieved evidence. If the evidence is insufficient, say so."
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
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.85)
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
            eos_token_id=tokenizer.eos_token_id,
            temperature=None if args.no_sample else args.temperature,
            top_p=None if args.no_sample else args.top_p,
            early_stopping=True,
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(answer)


if __name__ == "__main__":
    main()
