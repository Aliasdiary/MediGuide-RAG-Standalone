"""
MediGuide-RAG: a medical education and care-navigation RAG agent.
"""

import logging
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG, MediGuideConfig
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MediGuideRAGSystem:
    """Main class for the medical education and care-navigation RAG agent."""

    def __init__(self, config: MediGuideConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

    def initialize_system(self):
        print("正在初始化 MediGuide-RAG 医疗知识问答与风险控制系统...")
        self.data_module = DataPreparationModule(self.config.data_path)
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            base_url=self.config.ollama_base_url,
        )
        print("系统初始化完成。")

    def build_knowledge_base(self):
        print("\n正在构建 MedQuAD 医疗知识库...")
        print("加载结构化 MedQuAD 问答数据...")
        self.data_module.load_documents()
        print("构建问题检索块与完整问答父文档...")
        chunks = self.data_module.chunk_documents()
        index_manifest = {
            "dataset": "MedQuAD",
            "dataset_fingerprint": self.data_module.dataset_fingerprint(),
            "dataset_limit": self.config.dataset_limit,
            "dataset_seed": self.config.dataset_seed,
            "embedding_model": self.config.embedding_model,
            "chunk_strategy": "question-child/full-qa-parent-v1",
        }
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
            expected_manifest=index_manifest,
        )
        vectorstore = self.index_module.load_index()

        if vectorstore is None:
            print("未找到本地索引，开始构建 FAISS 向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            self.index_module.save_index()
        else:
            print("已加载本地 FAISS 向量索引。")

        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        print("\n知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   问题类型: {stats['question_types']}")
        print(f"   来源机构: {stats['sources']}")
        print(f"   UMLS 语义类型: {stats['semantic_types']}")
        print("知识库就绪。")

    def ask_question(self, question: str, stream: bool = False):
        if not all([self.retrieval_module, self.generation_module, self.data_module]):
            raise ValueError("请先构建知识库")

        print(f"\n用户问题: {question}")
        route_type = self.generation_module.query_router(question)
        print(f"查询路由: {route_type}")

        rewritten_query = self.generation_module.query_rewrite(question, route_type)
        if rewritten_query != question:
            print(f"查询改写: {rewritten_query}")

        filters = self._extract_filters_from_query(question, route_type)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters,
                top_k=self.config.top_k,
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        self._print_retrieval_hits(relevant_chunks)
        if not relevant_chunks:
            return "当前知识库中未找到相关信息。"

        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        if stream:
            return self.generation_module.generate_answer_stream(question, relevant_docs, route_type)
        return self.generation_module.generate_answer(question, relevant_docs, route_type)

    def _extract_filters_from_query(self, query: str, route_type: str) -> dict:
        question_types = {
            "triage": ["symptoms", "when to contact a medical professional", "diagnosis"],
            "education": [
                "information",
                "causes",
                "prevention",
                "susceptibility",
                "outlook",
                "inheritance",
                "frequency",
            ],
            "medication": [
                "treatment",
                "considerations",
                "information",
            ],
            "treatment": ["treatment", "exams and tests", "diagnosis", "management"],
        }
        return {"question_type": question_types.get(route_type, [])}

    def _print_retrieval_hits(self, chunks: List):
        if not chunks:
            print("检索命中: 0")
            return
        print(f"检索命中: {len(chunks)}")
        for index, chunk in enumerate(chunks, start=1):
            name = chunk.metadata.get("knowledge_name", "未知资料")
            question_type = chunk.metadata.get("question_type", "unknown")
            source_org = chunk.metadata.get("source_org", "Unknown")
            source_url = chunk.metadata.get("source_url", "")
            score = chunk.metadata.get("rrf_score", 0.0)
            print(f"   {index}. {name} | {source_org}/{question_type} | RRF={score:.4f}")
            print(f"      {source_url}")

    def run_interactive(self):
        print("=" * 60)
        print("MediGuide-RAG 医疗健康知识问答与风险控制 Agent")
        print("=" * 60)
        print("注意: 本系统仅用于健康科普与就医导诊，不能替代医生诊断。")
        print("示例: 白血病常见症状有哪些？")
        print("示例: 高血压有哪些常见风险因素？")
        print("示例: 胸痛伴呼吸困难是否需要立即就医？")
        print("输入 quit / exit 退出。")

        self.initialize_system()
        self.build_knowledge_base()

        while True:
            try:
                question = input("\n请输入问题: ").strip()
                if question.lower() in {"quit", "exit", ""}:
                    break
                print("\n回答:")
                for chunk in self.ask_question(question, stream=True):
                    print(chunk, end="", flush=True)
                print("\n")
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"处理问题时出错: {exc}")


def main():
    try:
        MediGuideRAGSystem().run_interactive()
    except Exception as exc:
        logger.error("系统运行失败: %s", exc)
        print(f"系统错误: {exc}")


if __name__ == "__main__":
    main()
