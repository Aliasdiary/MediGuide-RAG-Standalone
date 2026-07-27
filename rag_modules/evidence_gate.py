"""Evidence gating utilities for retrieval-aware medical generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document


EVIDENCE_SUFFICIENT = "sufficient"
EVIDENCE_PARTIAL = "partial"
EVIDENCE_INSUFFICIENT = "insufficient"
EVIDENCE_CONFLICT = "conflict"
EVIDENCE_SUBJECT_MISMATCH = "subject_mismatch"


@dataclass
class EvidenceGateResult:
    """Filtered evidence and its reliability status."""

    status: str
    usable_docs: List[Document]
    rejected_docs: List[Document]
    reasons: List[str]

    @property
    def has_usable_evidence(self) -> bool:
        return bool(self.usable_docs) and self.status in {EVIDENCE_SUFFICIENT, EVIDENCE_PARTIAL}


class EvidenceGate:
    """Rule-based evidence gate before RAG-SFT generation.

    The gate intentionally uses conservative lexical checks. It is not a clinical
    classifier; its job is to stop obviously wrong evidence from entering the
    generation prompt, especially high-ranked documents with the wrong subject.
    """

    QUESTION_TYPE_HINTS: Dict[str, Tuple[str, ...]] = {
        "medication": (
            "drug",
            "medicine",
            "medication",
            "dose",
            "dosage",
            "side effect",
            "antibiotic",
            "pill",
            "tablet",
            "药",
            "剂量",
            "加量",
            "副作用",
        ),
        "treatment": (
            "treatment",
            "therapy",
            "surgery",
            "test",
            "exam",
            "procedure",
            "治疗",
            "检查",
            "手术",
        ),
        "triage": (
            "symptom",
            "emergency",
            "urgent",
            "pain",
            "bleeding",
            "bite",
            "wound",
            "症状",
            "急诊",
            "疼",
            "出血",
            "咬",
            "伤口",
        ),
        "education": (
            "what is",
            "overview",
            "cause",
            "inherit",
            "disease",
            "什么是",
            "原因",
            "疾病",
        ),
    }

    HUMAN_ANIMAL_BITE_TERMS = (
        "dog bite",
        "animal bite",
        "rabies",
        "bite",
        "wound",
        "post-exposure",
        "post exposure",
        "我被狗咬",
        "被狗咬",
        "动物咬",
    )
    ANIMAL_MANAGEMENT_TERMS = (
        "euthanize",
        "quarantine",
        "isolate the animal",
        "livestock",
        "owner",
        "release after six months",
        "vaccinate your pet",
    )
    HUMAN_BITE_EVIDENCE_TERMS = (
        "person",
        "people",
        "human",
        "patient",
        "wash",
        "wound",
        "medical care",
        "post-exposure",
        "post exposure",
        "shots",
        "vaccine",
        "treatment",
    )

    DOSAGE_TERMS = (
        "dose",
        "dosage",
        "double",
        "increase",
        "adjust",
        "加倍",
        "加量",
        "剂量",
        "自行",
    )

    def __init__(self, min_support: int = 2, max_docs: int = 3):
        self.min_support = min_support
        self.max_docs = max_docs

    def assess(self, question: str, docs: Sequence[Document], route: str = "") -> EvidenceGateResult:
        question_text = question.casefold()
        route = route or self.infer_route(question_text)
        scored: List[Tuple[int, Document]] = []
        rejected: List[Document] = []
        reasons: List[str] = []

        for doc in docs:
            text = self._doc_text(doc)
            support, reason = self._support_score(question_text, text, doc, route)
            doc.metadata["evidence_support"] = support
            doc.metadata["evidence_gate_reason"] = reason

            if self._is_human_animal_bite(question_text) and self._is_animal_management_only(text):
                doc.metadata["evidence_status"] = EVIDENCE_SUBJECT_MISMATCH
                doc.metadata["evidence_gate_reason"] = "human_animal_bite_subject_mismatch"
                rejected.append(doc)
                reasons.append("filtered animal-management evidence for a human bite question")
                continue

            if support < self.min_support:
                doc.metadata["evidence_status"] = EVIDENCE_INSUFFICIENT
                rejected.append(doc)
                continue

            doc.metadata["evidence_status"] = EVIDENCE_SUFFICIENT if support >= self.min_support + 2 else EVIDENCE_PARTIAL
            scored.append((support, doc))

        scored.sort(
            key=lambda item: (
                item[0],
                float(item[1].metadata.get("rrf_score", 0.0)),
            ),
            reverse=True,
        )
        usable = self._dedupe_parents([doc for _, doc in scored])[: self.max_docs]

        if usable and any(doc.metadata.get("evidence_status") == EVIDENCE_SUFFICIENT for doc in usable):
            status = EVIDENCE_SUFFICIENT
        elif usable:
            status = EVIDENCE_PARTIAL
        elif rejected and any(doc.metadata.get("evidence_status") == EVIDENCE_SUBJECT_MISMATCH for doc in rejected):
            status = EVIDENCE_SUBJECT_MISMATCH
        else:
            status = EVIDENCE_INSUFFICIENT

        if not reasons:
            reasons.append(f"{len(usable)} usable evidence document(s)")
        return EvidenceGateResult(status=status, usable_docs=usable, rejected_docs=rejected, reasons=reasons)

    def infer_route(self, question: str) -> str:
        lowered = question.casefold()
        for route, terms in self.QUESTION_TYPE_HINTS.items():
            if any(term in lowered for term in terms):
                return route
        return "education"

    def dynamic_top_k(self, question: str, base_top_k: int = 4) -> int:
        lowered = question.casefold()
        if self._is_human_animal_bite(lowered) or any(term in lowered for term in self.DOSAGE_TERMS):
            return max(base_top_k, 6)
        if len(lowered) < 18:
            return max(base_top_k, 5)
        return base_top_k

    def _support_score(self, question: str, text: str, doc: Document, route: str) -> Tuple[int, str]:
        support = 0
        reasons: List[str] = []
        focus = str(doc.metadata.get("focus", "")).casefold()
        question_type = str(doc.metadata.get("question_type", "")).casefold()

        if focus and self._token_overlap(question, focus) > 0:
            support += 2
            reasons.append("focus_overlap")

        overlap = self._token_overlap(question, text)
        if overlap:
            support += min(3, overlap)
            reasons.append(f"token_overlap={overlap}")

        route_terms = self.QUESTION_TYPE_HINTS.get(route, ())
        route_hits = sum(1 for term in route_terms if term in text)
        if route_hits:
            support += min(2, route_hits)
            reasons.append(f"route_terms={route_hits}")

        if route == "medication" and any(term in text for term in ("drug", "medicine", "medication", "dose", "dosage", "药")):
            support += 2
            reasons.append("medication_match")

        if self._is_human_animal_bite(question) and any(term in text for term in self.HUMAN_BITE_EVIDENCE_TERMS):
            support += 3
            reasons.append("human_bite_evidence")

        if question_type and route in question_type:
            support += 1
            reasons.append("question_type_match")

        return support, ",".join(reasons) or "weak_match"

    @staticmethod
    def _doc_text(doc: Document) -> str:
        parts = [
            doc.page_content,
            str(doc.metadata.get("question", "")),
            str(doc.metadata.get("focus", "")),
            str(doc.metadata.get("question_type", "")),
            str(doc.metadata.get("semantic_type", "")),
        ]
        return "\n".join(parts).casefold()

    @classmethod
    def _is_human_animal_bite(cls, question: str) -> bool:
        return any(term in question for term in cls.HUMAN_ANIMAL_BITE_TERMS)

    @classmethod
    def _is_animal_management_only(cls, text: str) -> bool:
        animal_hits = sum(1 for term in cls.ANIMAL_MANAGEMENT_TERMS if term in text)
        human_hits = sum(1 for term in cls.HUMAN_BITE_EVIDENCE_TERMS if term in text)
        return animal_hits >= 1 and human_hits <= 1

    @staticmethod
    def _token_overlap(left: str, right: str) -> int:
        tokens = {
            token
            for token in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", left.casefold())
            if len(token) >= 2
        }
        if not tokens:
            return 0
        right_tokens = {
            token
            for token in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", right.casefold())
            if len(token) >= 2
        }
        return len(tokens.intersection(right_tokens))

    @staticmethod
    def _dedupe_parents(docs: Iterable[Document]) -> List[Document]:
        seen = set()
        unique: List[Document] = []
        for doc in docs:
            parent_id = doc.metadata.get("parent_id") or doc.metadata.get("record_id") or doc.page_content[:120]
            if parent_id in seen:
                continue
            seen.add(parent_id)
            unique.append(doc)
        return unique

