# MediGuide-RAG：基于可靠证据与安全约束的医疗问答风险控制 Agent

## 项目背景

针对大模型在医疗问答中容易出现事实幻觉、过度诊断和错误用药建议的问题，本项目构建
了一个本地医疗健康知识问答 Agent，通过可靠外部知识检索、跨语言混合召回、引用溯源
和安全生成约束降低越界回答风险。

## 数据来源与 ETL

知识库使用 CC BY 4.0 许可的 MedQuAD。该数据集总计 47,457 组问答，来源包括 NCI、
CDC、NIDDK、NINDS 等官方医疗网站。项目自动下载官方仓库并完成以下处理：

1. 校验 ZIP 格式和许可证文件，计算下载包 SHA256。
2. 解析包含有效答案的 1-9 子集，排除因 MedlinePlus 版权而移除答案的 10-12 子集。
3. 清洗 XML 空白字符，保留问题、答案、问题类型、主题、机构、URL 和 UMLS 元数据。
4. 按来源机构与问题类型分层，使用种子 42 确定性抽取 5,000 条。
5. 输出 JSONL 和 manifest，记录数据 SHA256、抽样配置和解析错误。

当前官方仓库实际解析得到 16,407 条有答案记录，抽样后的演示知识库为 5,000 条。

## Agent Workflow

```text
用户中文问题
-> triage / education / medication / treatment 意图路由
-> Qwen3 生成英文检索查询
-> BGE-M3 + FAISS 语义检索
-> BM25 英文关键词检索
-> RRF 融合重排
-> question_type 元数据过滤，无结果时回退混合检索
-> 命中问题小块后回填完整问答父文档
-> Qwen3 生成中文回答、标注机构和原始 URL
```

## 关键实现

**结构化数据处理：** 使用 Python 标准库完成下载、XML 解析、字段校验、清洗、分层抽样
与 JSONL 输出。索引 manifest 绑定数据指纹、抽样参数、Embedding 模型和分块策略，
配置变化后自动重建索引。

**跨语言混合检索：** 使用 BGE-M3 对英文 MedQuAD 问题建立 FAISS 索引；本地 Qwen3
将中文问题改写为英文检索式，使 BM25 和向量检索使用同一查询，并通过 RRF 融合排序。

**父子文档设计：** 检索块只保存 MedQuAD 问题，减少长答案对检索表征的干扰；生成阶段
根据 `parent_id` 回填完整问题、答案和来源字段，实现“小块检索、大块生成”。

**安全生成：** 危险信号规则独立于 MedQuAD 元数据，不虚构 `urgency_level`。胸痛、
呼吸困难、意识障碍等问题优先提示急救；Prompt 禁止确诊、处方和个体化剂量，并要求
证据不足时明确拒答。

**可解释输出：** CLI 展示命中主题、机构、问题类型、RRF 分数和原始 URL；最终回答使用
`[资料1]` 标注依据并列出资料来源。

## 项目边界

- 项目用于医疗健康科普和 RAG 工程研究，不承担诊断或治疗决策。
- MedQuAD 是英文医疗资料，回答由本地模型生成中文摘要，原始 URL 是最终核验依据。
- 版权受限的 MedlinePlus 药物答案未纳入，因此不宣称覆盖完整药品说明书或具体剂量。
- 知识库不包含中国本地挂号、医保和医院流程，因此不提供相关能力。
- 当前未进行临床有效性评估，不声明未经实验支持的准确率提升。

## 演示问题与预期

| Query | Route | Expected evidence |
|---|---|---|
| 白血病常见症状有哪些？ | triage | symptoms |
| 高血压有哪些常见风险因素？ | education | causes / susceptibility |
| 哮喘通常通过哪些检查诊断？ | treatment | exams and tests / diagnosis |
| 癫痫有哪些常见治疗方式？ | treatment | treatment |
| 糖尿病可能有哪些并发症？ | education | information / complications |
| 某种疾病的治疗通常需要考虑什么？ | medication | treatment / considerations |
| 哪些人更容易患乳腺癌？ | education | susceptibility |
| 如何预防流感？ | education | prevention |
| 帕金森病的预后如何？ | education | outlook |
| 胸痛伴呼吸困难是否需要立即就医？ | triage | danger rule first, then evidence |

## 简历描述

**基于可靠证据与安全约束的医疗问答风险控制 Agent**

- 接入 MedQuAD 开放医疗问答数据集，设计 XML ETL、字段标准化、确定性分层抽样和来源追踪流程，构建 5,000 条可溯源本地知识库。
- 使用 BGE-M3 + FAISS 与 BM25 实现中英跨语言混合检索，通过 RRF 融合召回和父文档回填增强证据完整性。
- 基于 LangChain 编排意图路由、英文查询改写、元数据过滤、安全生成和引用溯源 Workflow。
- 使用 Ollama 本地 `qwen3:latest` 生成中文答案，通过危险信号规则与 Prompt 约束降低诊断、处方及个体化剂量等越界风险。
