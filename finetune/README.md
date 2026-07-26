# MediGuide-SFT / QLoRA Fine-Tuning

This folder turns the existing MedQuAD project into a standalone SFT/QLoRA
workflow. The recommended AutoDL setup is:

- GPU: RTX 4090 24GB
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: QLoRA, 4-bit NF4
- Trainer: LLaMA-Factory

The fine-tuning goal is not to memorize all medical facts. It teaches the model
to produce safer medical education answers, cite source metadata, refuse unsafe
requests, and handle emergency-signal questions.

## 1. Prepare Environment On AutoDL

Choose an official PyTorch base image instead of a community image. The base
environment should already contain working PyTorch and should not contain vLLM.

```text
Ubuntu 22.04
PyTorch 2.2/2.3
CUDA 12.1
```

Do not install packages into `base`. Only inspect it:

```bash
nvidia-smi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

pip show vllm
```

If `pip show vllm` says the package is not found, clone `base` into the training
environment:

```bash
conda create -n med-sft --clone base -y
conda activate med-sft
```

Install LLaMA-Factory in `med-sft` only. Do not use broad `pip install -U
transformers datasets ...` commands, because they can upgrade the stack beyond
LLaMA-Factory's supported range.

```bash
pip install llamafactory==0.9.5 \
  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

pip check
llamafactory-cli env
llamafactory-cli version
```

Keep vLLM out of `med-sft`. After training, create a separate `med-vllm`
environment for serving and batch inference.

## 2. Clone This Project

```bash
cd /root/autodl-tmp
git clone https://github.com/Aliasdiary/MediGuide-RAG-Standalone.git
cd MediGuide-RAG-Standalone
```

The repository already contains a 5,000-record MedQuAD subset. To prepare a
full effective MedQuAD file on AutoDL, run:

```bash
python scripts/prepare_medquad.py --limit 0 --output data/medquad_full.jsonl
```

`--limit 0` means "keep all valid records from the allowed MedQuAD subsets".

## 3. Download Qwen2.5-3B-Instruct

Prefer ModelScope in China:

```bash
modelscope download --model Qwen/Qwen2.5-3B-Instruct \
  --local_dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

Or use Hugging Face:

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

Ollama models such as `qwen3:latest` are GGUF/quantized inference artifacts and
are not used as SFT training bases.

## 4. Build SFT Dataset

Use the included 5,000-record subset:

```bash
python finetune/build_sft_dataset.py \
  --input data/medquad_5000.jsonl \
  --output-dir finetune/llamafactory_data \
  --train-size 4500 \
  --valid-size 300 \
  --seed 42
```

Use the full effective MedQuAD file:

```bash
python finetune/build_sft_dataset.py \
  --input data/medquad_full.jsonl \
  --output-dir finetune/llamafactory_data \
  --train-size 14000 \
  --valid-size 1000 \
  --seed 42
```

The script creates:

```text
finetune/llamafactory_data/
├── dataset_info.json
├── mediguide_sft_train.jsonl
├── mediguide_sft_valid.jsonl
└── mediguide_sft_manifest.json
```

## 5. Train QLoRA With LLaMA-Factory

Run the training config by absolute path:

```bash
llamafactory-cli train /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/qwen25_3b_qlora_sft.yaml
```

If CUDA memory is tight, reduce `cutoff_len` from `2048` to `1536` or reduce
`lora_rank` from `16` to `8`.

## 6. Export And Inference

The first useful artifact is the LoRA adapter under:

```text
/root/autodl-tmp/MediGuide-RAG-Standalone/finetune/output/qwen25-3b-mediguide-qlora
```

Merge the LoRA adapter into the base model before standalone inference:

```bash
llamafactory-cli export /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export_qwen25_3b_mediguide.yaml
```

The exported full model is written to:

```text
/root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export/qwen25-3b-mediguide-sft
```

If vLLM is unavailable or the server environment has CUDA/cuDNN version
conflicts, run the exported model with Transformers:

```bash
python finetune/infer_sft.py --question "我能不能直接把降压药剂量加倍？"
python finetune/infer_sft.py --question "突然胸痛并且呼吸困难，可以明天再去医院吗？"
```

To connect the RAG pipeline with the SFT model at inference time, use:

```bash
python finetune/infer_rag_sft.py --question "我能不能直接把降压药剂量加倍？"
```

This command first retrieves MedQuAD evidence with the existing BGE-M3, FAISS,
BM25, and RRF pipeline, backfills parent QA documents, and then passes the
retrieved evidence plus the user question to the exported MediGuide-SFT model.
Use this path when demonstrating RAG-grounded SFT generation.

vLLM acceleration is optional. Do it after training in a separate environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n med-vllm --clone base -y
conda activate med-vllm

pip install vllm openai pandas tqdm \
  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

Then serve the exported model with vLLM if the dependency stack is compatible.
Do not run vLLM and training at the same time on one RTX 4090, because they
will compete for GPU memory.

## 7. Evaluate SFT Generation Quality

Compare the base model and the exported MediGuide-SFT model on a fixed Chinese
medical safety generation set:

```bash
python finetune/evaluate_sft.py --limit 5
python finetune/evaluate_sft.py
```

The default SFT generation length is `--max-new-tokens 192`, which is intended
to complete the four-line safety format without encouraging long disclaimer
drift. You can adjust it for debugging:

```bash
python finetune/infer_sft.py --question "..." --max-new-tokens 256
python finetune/evaluate_sft.py --max-new-tokens 256
```

The script writes:

```text
finetune/eval_results/
├── sft_predictions.jsonl
├── sft_metrics.json
└── sft_report.md
```

Metrics include safety compliance, medication safety, emergency awareness,
format compliance, hallucination control, and keyword coverage. They are
rule-based engineering checks and do not represent clinical accuracy.

## Recommended Resume Positioning

MediGuide-SFT uses MedQuAD official medical QA records to build an instruction
fine-tuning dataset, then applies QLoRA to Qwen2.5-3B-Instruct on an RTX 4090.
The model is optimized for medical education answers, safety refusal, emergency
triage-style warnings, and source-aware response formatting.
