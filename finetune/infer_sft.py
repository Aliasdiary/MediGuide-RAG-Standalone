"""Run the exported MediGuide SFT model with Transformers.

This script matches the minimal inference path verified on AutoDL:
manual Qwen ChatML prompt, greedy generation, and the tokenizer's default EOS.
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
)


def build_qwen_chatml_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with exported MediGuide SFT model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
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
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(answer)


if __name__ == "__main__":
    main()
