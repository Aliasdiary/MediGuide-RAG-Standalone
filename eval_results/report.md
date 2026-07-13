# MediGuide-RAG Evaluation Report

- Generated at: 2026-07-10T14:15:48
- Cases: 40
- LLM: `qwen3:latest`
- Embedding: `BAAI/bge-m3`

| Metric | LLM-only | MediGuide-RAG | Gain |
| --- | ---: | ---: | ---: |
| Evidence Hit@4 | 0.0% | 87.5% | +87.5 pp |
| Citation Rate | 7.5% | 97.5% | +90.0 pp |
| Keyword Coverage | 97.5% | 97.1% | -0.4 pp |
| Safety Compliance | 97.5% | 100.0% | +2.5 pp |
| Unsupported Refusal Rate | 97.5% | 100.0% | +2.5 pp |
| Composite | 75.0% | 98.7% | +23.6 pp |

## Resume-Friendly Summary

在自建 40 条中文医疗问答评测集上，相比 LLM-only，MediGuide-RAG 在综合指标上提升 +23.6 个百分点；其中来源引用率提升 +90.0 个百分点，关键词覆盖率提升 -0.4 个百分点，安全合规率提升 +2.5 个百分点。

## Notes

- 该评测用于工程对比，不代表临床诊断准确率。
- LLM-only 不使用 MedQuAD、FAISS、BM25、RRF 或父文档回填。
- MediGuide-RAG 使用当前完整流程，并要求回答给出来源机构和原始 URL。
