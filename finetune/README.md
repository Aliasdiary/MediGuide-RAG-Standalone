# MediGuide Fine-Tuning

本目录包含两类微调数据与推理脚本：

- `build_sft_dataset.py`：原始 MedQuAD 医疗科普 SFT 数据，用于学习基础问答风格和安全边界。
- `build_retrieval_aware_sft_dataset.py`：Retrieval-Aware SFT 数据，把真实 RAG 召回证据、hard negative、证据不足和主体错配样本加入训练输入。
- `infer_sft.py`：纯 SFT 模型推理，用于基线对比。
- `infer_rag_sft.py`：RAG 检索证据进入 SFT 模型的统一推理入口。

## Environment

推荐 AutoDL RTX 4090 24GB：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n med-sft --clone base -y
conda activate med-sft
pip install llamafactory==0.9.5 -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip check
```

下载基座模型：

```bash
modelscope download --model Qwen/Qwen2.5-3B-Instruct \
  --local_dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

下载 BGE-M3：

```bash
modelscope download --model BAAI/bge-m3 \
  --local_dir /root/autodl-tmp/models/bge-m3
```

## Build Retrieval-Aware Data

```bash
cd /root/autodl-tmp/MediGuide-RAG-Standalone
python finetune/build_retrieval_aware_sft_dataset.py \
  --input data/medquad_5000.jsonl \
  --output-dir finetune/retrieval_aware_data \
  --train-size 4000 \
  --valid-size 300 \
  --embedding-model /root/autodl-tmp/models/bge-m3
```

输出：

```text
finetune/retrieval_aware_data/
├── dataset_info.json
├── mediguide_retrieval_sft_train.jsonl
├── mediguide_retrieval_sft_valid.jsonl
└── manifest.json
```

## Train QLoRA

```bash
llamafactory-cli train /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/qwen25_3b_qlora_retrieval_sft.yaml
```

训练策略：

- 基座模型：`Qwen/Qwen2.5-3B-Instruct`
- 方法：QLoRA，4-bit，LoRA rank 16
- `cutoff_len`: 3072，用于容纳用户问题与检索证据
- `learning_rate`: `1.5e-4`
- `num_train_epochs`: `2.0`

## Export

```bash
llamafactory-cli export /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export_qwen25_3b_mediguide.yaml
```

导出后通过统一 RAG-SFT 入口推理：

```bash
python finetune/infer_rag_sft.py \
  --question "我被狗咬了怎么办？" \
  --embedding-model /root/autodl-tmp/models/bge-m3 \
  --model-path /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export/qwen25-3b-mediguide-sft
```

