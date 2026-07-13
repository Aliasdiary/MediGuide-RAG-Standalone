"""Routing, cross-lingual rewriting, and safe generation for MediGuide-RAG."""

from __future__ import annotations

import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """Handle medical intent routing, English retrieval queries, and cited answers."""

    VALID_ROUTES = {"triage", "education", "medication", "treatment"}
    DANGER_SIGNALS = [
        "胸痛",
        "呼吸困难",
        "意识障碍",
        "昏迷",
        "抽搐",
        "严重过敏",
        "持续高热",
        "大出血",
        "自杀",
        "chest pain",
        "difficulty breathing",
        "unconscious",
        "seizure",
        "severe bleeding",
    ]
    QUERY_SYNONYMS = {
        "hypertension": "high blood pressure",
        "hypotension": "low blood pressure",
        "dyspnea": "shortness of breath",
        "myocardial infarction": "heart attack",
        "cerebrovascular accident": "stroke",
        "cephalalgia": "headache",
    }

    def __init__(
        self,
        model_name: str = "qwen3:latest",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.llm = None
        self.route_llm = None
        self.rewrite_llm = None
        self.setup_llm()

    def setup_llm(self):
        common = {
            "model": self.model_name,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "reasoning": False,
        }
        self.llm = ChatOllama(num_predict=self.max_tokens, **common)
        self.route_llm = ChatOllama(num_predict=32, **common)
        self.rewrite_llm = ChatOllama(num_predict=128, **common)

    def query_router(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            """
/no_think
你是医疗健康知识问答系统的查询路由器。将用户问题分类为以下一种：
triage：描述症状，希望了解危险信号、严重程度或是否需要及时就医。
education：询问疾病定义、原因、风险因素、预防、预后或一般医学知识。
medication：询问药物用途、副作用、禁忌或用药安全。
treatment：询问诊断检查、治疗方法、治疗副作用或疾病管理。

不要输出思考过程。只返回 triage、education、medication 或 treatment。
用户问题：{query}
"""
        )
        chain = ({"query": RunnablePassthrough()} | prompt | self.route_llm | StrOutputParser())
        result = self._clean_response(chain.invoke(query)).strip().lower()
        return result if result in self.VALID_ROUTES else self._rule_based_route(query)

    def _rule_based_route(self, query: str) -> str:
        lowered = query.lower()
        if any(signal in lowered for signal in self.DANGER_SIGNALS):
            return "triage"
        if any(word in lowered for word in ["药", "用药", "抗生素", "副作用", "禁忌", "medicine", "drug"]):
            return "medication"
        if any(word in lowered for word in ["检查", "诊断", "治疗", "手术", "化疗", "test", "diagnos", "treat"]):
            return "treatment"
        if any(word in lowered for word in ["发烧", "疼痛", "咳嗽", "头晕", "症状", "严重吗", "symptom"]):
            return "triage"
        return "education"

    def query_rewrite(self, query: str, route_type: str) -> str:
        prompt = PromptTemplate(
            template="""
/no_think
Rewrite the user's medical question as one concise English search query for the English MedQuAD
knowledge base. Preserve disease names, symptoms, medicines, duration, age, and the original intent.
Do not answer the question. Do not add a diagnosis. Output only the English search query.
The output must be English and must not contain Chinese characters.
When a condition has both a clinical term and a common English name, include both without explanation,
for example "hypertension high blood pressure risk factors".

Intent: {route_type}
User question: {query}
English retrieval query:
""",
            input_variables=["query", "route_type"],
        )
        chain = (
            {"query": lambda value: value["query"], "route_type": lambda value: value["route_type"]}
            | prompt
            | self.rewrite_llm
            | StrOutputParser()
        )
        rewritten = self._clean_response(chain.invoke({"query": query, "route_type": route_type})).strip()
        if self._contains_cjk(rewritten):
            fallback_prompt = ChatPromptTemplate.from_template(
                """
/no_think
Translate the following Chinese medical search query into concise English.
Return only the English translation. Do not use any Chinese characters.
Query: {query}
"""
            )
            fallback_chain = fallback_prompt | self.rewrite_llm | StrOutputParser()
            fallback = self._clean_response(fallback_chain.invoke({"query": query})).strip()
            if fallback and not self._contains_cjk(fallback):
                rewritten = fallback
        return self._expand_medical_synonyms(rewritten or query)

    def contains_danger_signal(self, query: str) -> bool:
        lowered = query.lower()
        return any(signal in lowered for signal in self.DANGER_SIGNALS)

    def generate_answer(self, query: str, context_docs: List[Document], route_type: str) -> str:
        prompt = self._answer_prompt()
        values = {
            "question": query,
            "route_type": route_type,
            "danger_signal": "yes" if self.contains_danger_signal(query) else "no",
            "context": self._build_context(context_docs),
        }
        chain = prompt | self.llm | StrOutputParser()
        return self._clean_response(chain.invoke(values))

    def generate_answer_stream(self, query: str, context_docs: List[Document], route_type: str):
        values = {
            "question": query,
            "route_type": route_type,
            "danger_signal": "yes" if self.contains_danger_signal(query) else "no",
            "context": self._build_context(context_docs),
        }
        chain = self._answer_prompt() | self.llm | StrOutputParser()
        for chunk in chain.stream(values):
            yield chunk

    @staticmethod
    def _answer_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """
/no_think
你是 MediGuide-RAG 医疗健康知识问答与风险控制助手。请用中文回答，并严格依据给出的
MedQuAD 资料。不得把资料转化为对用户的确诊、处方或个体化药物剂量。

用户问题：{question}
问题类型：{route_type}
检测到危险信号：{danger_signal}

MedQuAD 检索资料：
{context}

要求：
1. 开头简要回答问题，然后列出关键依据。
2. 每项医学结论使用 [资料1] 形式标明依据，末尾列出“资料来源”，包含机构和原始 URL。
3. 如果危险信号为 yes，先建议尽快联系当地急救服务或前往急诊，不等待在线回答。
4. 不诊断具体疾病，不开具处方，不给出针对个人的剂量；药物问题建议咨询医生或药师。
5. 如果资料不足，明确写“当前知识库中未找到足够信息”，不要用模型常识补全。
6. 结尾说明回答仅用于健康科普，不能替代医生诊断。
不要输出思考过程或 <think> 标签。
"""
        )

    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u3400-\u9fff]", text or ""))

    @classmethod
    def _expand_medical_synonyms(cls, query: str) -> str:
        lowered = query.lower()
        additions = []
        for clinical_term, common_term in cls.QUERY_SYNONYMS.items():
            if clinical_term in lowered and common_term not in lowered:
                additions.append(common_term)
            elif common_term in lowered and clinical_term not in lowered:
                additions.append(clinical_term)
        return " ".join([query, *additions]).strip()

    @staticmethod
    def _build_context(docs: List[Document], max_length: int = 7000) -> str:
        if not docs:
            return "当前知识库中未找到相关信息。"
        parts = []
        current_length = 0
        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            header = (
                f"[资料{index}] {metadata.get('focus', 'Unknown topic')}\n"
                f"机构: {metadata.get('source_org', 'Unknown')}\n"
                f"问题类型: {metadata.get('question_type', 'unknown')}\n"
                f"UMLS: {metadata.get('umls_cui', '')}\n"
                f"原始URL: {metadata.get('source_url', '')}\n"
                f"许可证: {metadata.get('license', 'CC BY 4.0')}"
            )
            text = f"{header}\n{doc.page_content}\n"
            if current_length + len(text) > max_length:
                break
            parts.append(text)
            current_length += len(text)
        return "\n\n" + ("\n" + "=" * 50 + "\n").join(parts)
