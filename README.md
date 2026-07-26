# MediGuide-RAG Standalone

这是从原 RAG 工程中完整提取的独立项目，可在当前目录直接运行。项目使用 MedQuAD
可靠医疗问答数据、BGE-M3、FAISS、BM25、RRF 和本地 Ollama Qwen3，实现中文医疗
知识问答、风险提示与来源追溯。

> 本项目仅用于健康科普和研究，不能替代医生诊断、处方或急救服务。

## 目录

```text
MediGuide-RAG-Standalone/
├── run.py                     # CLI 入口
├── config.py                  # 独立路径与模型配置
├── requirements.txt
├── rag_modules/               # ETL 后的数据加载、索引、检索和生成模块
├── scripts/
│   └── prepare_medquad.py     # MedQuAD 下载、清洗和抽样
├── tests/
├── data/
│   ├── medquad_5000.jsonl
│   └── medquad_5000.manifest.json
├── medical_vector_index/      # 已构建 FAISS 索引
├── docs/
└── THIRD_PARTY_NOTICES.md
```

数据路径和索引路径均根据 `config.py` 所在位置计算，因此可以从任意工作目录启动，
不依赖原仓库中的其他代码或数据。

## 环境要求

- Python 3.10 或 3.11
- Ollama
- 本地模型 `qwen3:latest`
- 首次安装依赖或重新下载 BGE-M3 时需要网络

## 安装与运行

```powershell
cd MediGuide-RAG-Standalone
pip install -r requirements.txt
ollama pull qwen3:latest
python run.py
```

项目已经包含 5,000 条 MedQuAD 数据和对应 FAISS 索引。BGE-M3 首次加载时仍需存在于
Hugging Face 本地缓存；如果没有，程序会自动下载。

## 重新准备数据

```powershell
python scripts\prepare_medquad.py --limit 5000 --seed 42
```

该命令会从 MedQuAD 官方仓库下载 XML，校验许可证，排除无答案子集，重新生成 JSONL
和数据 manifest。数据变化后，下一次运行会根据指纹自动重建 FAISS 索引。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## Workflow

```text
中文问题
→ triage / education / medication / treatment 意图路由
→ Qwen3 英文查询改写
→ 医学同义词扩展
→ BGE-M3 + FAISS / BM25
→ RRF 融合与 focus 主题加权
→ question_type 元数据过滤
→ 完整问答父文档回填
→ 危险信号与安全 Prompt
→ 中文回答、来源机构和原始 URL
```

项目设计和简历说明见
`docs/mediguide_project_description.md`。

## Evaluation

The project includes a reproducible binary evaluation between `LLM-only` and
the full `MediGuide-RAG` workflow.

```powershell
python scripts\evaluate.py --limit 10
python scripts\evaluate.py
```

Outputs are written to `eval_results/`:

- `metrics.json`: aggregate metrics and percentage-point gains.
- `predictions.jsonl`: per-question answers, retrieval hits, and scores.
- `report.md`: resume-friendly comparison table.

The current 40-case run compares local `qwen3:latest` without RAG against the
full MedQuAD + BGE-M3 + FAISS/BM25 + RRF + citation workflow. The metrics are
engineering evaluation numbers only and do not represent clinical diagnostic
accuracy.

## RAG-grounded SFT / QLoRA Fine-Tuning

The project also includes a RAG-grounded SFT path for AutoDL RTX 4090 experiments.
It builds Alpaca-style medical instruction data from MedQuAD where each sample contains
a user question plus retrieved evidence/source metadata, then fine-tunes
`Qwen/Qwen2.5-3B-Instruct` with LLaMA-Factory QLoRA.

```bash
python finetune/build_sft_dataset.py \
  --input data/medquad_5000.jsonl \
  --output-dir finetune/llamafactory_data \
  --train-size 4500 \
  --valid-size 300
```

On AutoDL:

```bash
cd /root/autodl-tmp
git clone https://github.com/Aliasdiary/MediGuide-RAG-Standalone.git

modelscope download --model Qwen/Qwen2.5-3B-Instruct \
  --local_dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct

cd /root/autodl-tmp/MediGuide-RAG-Standalone
python finetune/build_sft_dataset.py \
  --input data/medquad_5000.jsonl \
  --output-dir finetune/llamafactory_data \
  --train-size 4500 \
  --valid-size 300

llamafactory-cli train /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/qwen25_3b_qlora_sft.yaml
```

See `finetune/README.md` for the full AutoDL environment setup and training
notes, including the recommended `base -> med-sft` cloned environment and a
separate later `med-vllm` inference environment. Ollama models remain useful
for the RAG generation baseline, but they are not used as SFT training bases.
