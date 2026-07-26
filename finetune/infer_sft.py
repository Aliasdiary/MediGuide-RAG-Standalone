"""Run the exported MediGuide SFT model with Transformers.

This is the stable fallback path when vLLM is unavailable in a constrained
AutoDL environment. It expects the LoRA adapter to have been merged/exported
with `finetune/export_qwen25_3b_mediguide.yaml`.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/MediGuide-RAG-Standalone/"
    "finetune/export/qwen25-3b-mediguide-sft"
)

SAFETY_SYSTEM_PROMPT = (
    "你是医疗健康科普助手，只提供健康科普信息，不进行诊断、不提供处方或个体化剂量。"
    "当问题涉及胸痛、呼吸困难、意识障碍、严重过敏等危险信号时，应优先建议及时就医或急诊。"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with exported MediGuide SFT model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
            repetition_penalty=1.15,
            no_repeat_ngram_size=8,
            eos_token_id=qwen_eos_token_ids(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    answer = answer.split("<|im_end|>", 1)[0].strip()
    print(answer)


if __name__ == "__main__":
    main()
