# MediGuide-RAG Standalone

**基于混合检索与 Retrieval-Aware SFT 的医疗健康问答 Agent**

本项目面向医疗健康科普与风险提示场景，构建一个可本地运行的医疗问答 Agent。系统通过 MedQuAD 可靠医疗知识库、跨语言混合检索、证据门控、Retrieval-Aware SFT 和医疗安全校验，提升回答的可追溯性、证据一致性和安全边界。

> 本项目仅用于健康科普、RAG/SFT 工程研究与简历展示，不能替代医生诊断、处方、个体化剂量建议或急救服务。

## Project Structure

```text
MediGuide-RAG-Standalone/
├── run.py                         # 原 RAG CLI 入口，保留 Ollama Qwen3 生成链路
├── config.py                      # 数据、索引、Embedding、RAG-SFT 参数
├── rag_modules/
│   ├── data_preparation.py        # MedQuAD JSONL 读取、父子文档建模
│   ├── index_construction.py      # BGE-M3 Embedding + FAISS 本地索引
│   ├── retrieval_optimization.py  # FAISS + BM25 + RRF + 可选 Cross-Encoder reranker
│   ├── evidence_gate.py           # 证据门控、主体一致性、动态 Top-K
│   └── generation_integration.py  # 原 Ollama RAG 生成模块
├── scripts/
│   ├── prepare_medquad.py         # MedQuAD 下载、XML 解析、清洗、抽样
│   └── evaluate.py                # RAG vs LLM-only 工程评估
├── finetune/
│   ├── build_sft_dataset.py       # 原医疗科普 SFT 数据
│   ├── build_retrieval_aware_sft_dataset.py
│   ├── infer_sft.py               # 纯 SFT 推理
│   ├── infer_rag_sft.py           # RAG 检索证据 + SFT 统一生成
│   ├── evaluate_sft.py            # Base vs SFT 评估
│   └── qwen25_3b_qlora_retrieval_sft.yaml
├── data/
│   ├── medquad_5000.jsonl
│   └── medquad_5000.manifest.json
├── medical_vector_index/          # FAISS Flat 本地索引与 manifest
└── docs/
```

## Architecture

```text
用户中文问题
-> 风险/意图识别与双语查询扩展
-> BGE-M3 + FAISS 语义检索
-> BM25 关键词检索
-> RRF 融合排序
-> 可选 Cross-Encoder Reranker
-> Parent Document 去重与回填
-> Evidence Gate 证据门控
-> Retrieval-Aware Qwen2.5-3B SFT 生成
-> 生成后安全与格式校验
```

## Key Design

- **MedQuAD ETL**：从公开 MedQuAD 数据集中解析 XML，完成清洗、分层抽样、JSONL 转换和版本管理，保留 `focus`、`question_type`、`source_org`、`source_url`、`semantic_type` 等元数据。
- **父子文档建模**：完整问答作为 Parent Document，问题文本作为 Child Chunk，检索阶段命中短问题，生成阶段回填完整问答上下文。
- **跨语言混合检索**：使用 BGE-M3 处理中英文语义匹配，结合 FAISS 和 BM25 多路召回，并用 RRF 融合排名。
- **证据门控**：`EvidenceGate` 对候选证据做主体一致性、问题类型、医学关键词和风险场景检查，输出 `sufficient`、`partial`、`insufficient`、`subject_mismatch` 等状态。
- **Retrieval-Aware SFT**：用真实 RAG 召回结果构造训练输入，让 Qwen2.5-3B-Instruct 学习在候选证据中筛选、拒绝错配证据，并生成更符合医疗科普边界的回答。

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

如果使用原 RAG 生成链路，需要本地 Ollama：

```bash
ollama pull qwen3:latest
python run.py
```

如果使用 RAG-SFT 统一推理，需要先准备 Qwen2.5-3B SFT 合并模型：

```bash
python finetune/infer_rag_sft.py \
  --question "我能不能直接把降压药剂量加倍？" \
  --embedding-model /root/autodl-tmp/models/bge-m3 \
  --model-path /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export/qwen25-3b-mediguide-sft
```

## Retrieval-Aware SFT

构造检索感知训练数据：

```bash
python finetune/build_retrieval_aware_sft_dataset.py \
  --input data/medquad_5000.jsonl \
  --output-dir finetune/retrieval_aware_data \
  --train-size 4000 \
  --valid-size 300 \
  --embedding-model /root/autodl-tmp/models/bge-m3
```

在 AutoDL RTX 4090 上训练：

```bash
llamafactory-cli train /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/qwen25_3b_qlora_retrieval_sft.yaml
```

导出合并模型：

```bash
llamafactory-cli export /root/autodl-tmp/MediGuide-RAG-Standalone/finetune/export_qwen25_3b_mediguide.yaml
```

## Evaluation

RAG 主链路评估：

```bash
python scripts/evaluate.py
```

SFT 基线评估：

```bash
python finetune/evaluate_sft.py
```

文本污染扫描：

```bash
python finetune/scan_text_contamination.py finetune data
```

已有基线结果：

- MediGuide-SFT 相比 Qwen2.5-3B-Instruct，医学关键词覆盖率由 **67.9%** 提升至 **74.6%**。
- 急症风险提示率由 **95.0%** 提升至 **100.0%**。
- 综合可用性由 **77.1%** 提升至 **79.1%**。
- 原 RAG 系统相比 LLM-only，综合可用性由 **75.0%** 提升至 **98.7%**。

