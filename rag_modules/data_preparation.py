"""MedQuAD data preparation for the medical RAG agent."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """Load structured MedQuAD records and create question-parent document pairs."""

    REQUIRED_FIELDS = {
        "question",
        "answer",
        "question_type",
        "focus",
        "source_org",
        "source_url",
        "umls_cui",
        "semantic_type",
        "license",
    }

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.data_files: List[Path] = []

    def load_documents(self) -> List[Document]:
        root = Path(self.data_path)
        self.data_files = sorted(root.glob("medquad_*.jsonl"))
        if not self.data_files:
            raise FileNotFoundError(
                f"No prepared MedQuAD JSONL found in {root}. "
                "Run: python scripts/prepare_medquad.py --limit 5000 --seed 42"
            )

        documents = []
        for data_file in self.data_files:
            with data_file.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        missing = self.REQUIRED_FIELDS.difference(record)
                        if missing:
                            raise ValueError(f"missing fields: {sorted(missing)}")
                        if not record["question"].strip() or not record["answer"].strip():
                            raise ValueError("question and answer must not be empty")
                        documents.append(self._record_to_document(record, data_file))
                    except (json.JSONDecodeError, ValueError) as exc:
                        logger.warning("Skipping %s:%s: %s", data_file, line_number, exc)

        if not documents:
            raise ValueError("Prepared MedQuAD files contain no valid records")
        self.documents = documents
        logger.info("Loaded %s MedQuAD parent documents", len(documents))
        return documents

    @staticmethod
    def _record_to_document(record: Dict[str, Any], data_file: Path) -> Document:
        record_id = str(record.get("id") or hashlib.sha256(record["question"].encode("utf-8")).hexdigest())
        parent_id = hashlib.sha256(f"{record['source_url']}|{record_id}".encode("utf-8")).hexdigest()
        focus = record["focus"].strip() or record["question"].strip()
        content = f"# {focus}\n\n## Question\n{record['question'].strip()}\n\n## Answer\n{record['answer'].strip()}"
        metadata = {
            "record_id": record_id,
            "parent_id": parent_id,
            "doc_type": "parent",
            "knowledge_name": focus,
            "question": record["question"].strip(),
            "question_type": record["question_type"].strip().lower(),
            "focus": focus,
            "source_org": record["source_org"].strip(),
            "source_url": record["source_url"].strip(),
            "umls_cui": record["umls_cui"].strip(),
            "semantic_type": record["semantic_type"].strip(),
            "source_subset": str(record.get("source_subset", "")).strip(),
            "license": record["license"].strip(),
            "source": str(data_file),
        }
        return Document(page_content=content, metadata=metadata)

    def chunk_documents(self) -> List[Document]:
        if not self.documents:
            raise ValueError("Load documents before chunking")

        chunks = []
        for doc in self.documents:
            child_id = hashlib.sha256(f"{doc.metadata['parent_id']}|question".encode("utf-8")).hexdigest()
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "chunk_id": child_id,
                    "doc_type": "child",
                    "chunk_index": 0,
                    "chunk_size": len(doc.metadata["question"]),
                }
            )
            chunk = Document(page_content=doc.metadata["question"], metadata=metadata)
            self.parent_child_map[child_id] = doc.metadata["parent_id"]
            chunks.append(chunk)

        self.chunks = chunks
        logger.info("Created %s MedQuAD question chunks", len(chunks))
        return chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        parent_by_id = {doc.metadata["parent_id"]: doc for doc in self.documents}
        seen = set()
        parents = []
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id and parent_id not in seen and parent_id in parent_by_id:
                seen.add(parent_id)
                parents.append(parent_by_id[parent_id])
        return parents

    def dataset_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for data_file in self.data_files:
            digest.update(data_file.name.encode("utf-8"))
            digest.update(data_file.read_bytes())
        return digest.hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        question_types: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        semantic_types: Dict[str, int] = {}
        for doc in self.documents:
            question_type = doc.metadata.get("question_type", "unknown")
            source_org = doc.metadata.get("source_org", "unknown")
            semantic_type = doc.metadata.get("semantic_type", "unknown")
            question_types[question_type] = question_types.get(question_type, 0) + 1
            sources[source_org] = sources.get(source_org, 0) + 1
            semantic_types[semantic_type] = semantic_types.get(semantic_type, 0) + 1
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "question_types": question_types,
            "sources": sources,
            "semantic_types": semantic_types,
        }
