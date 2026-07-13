"""
Vector index construction module for MediGuide-RAG.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """Builds and loads a local FAISS vector index."""

    MANIFEST_FILE = "index_manifest.json"

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        index_save_path: str = "./medical_vector_index",
        expected_manifest: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.expected_manifest = expected_manifest or {}
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        logger.info("Initializing embedding model: %s", self.model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        if not chunks:
            raise ValueError("Document chunks cannot be empty")
        self.vectorstore = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        logger.info("Built FAISS index with %s chunks", len(chunks))
        return self.vectorstore

    def save_index(self):
        if not self.vectorstore:
            raise ValueError("Build or load the vector index first")
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        manifest_path = Path(self.index_save_path) / self.MANIFEST_FILE
        manifest_path.write_text(
            json.dumps(self.expected_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved vector index to %s", self.index_save_path)

    def load_index(self):
        index_path = Path(self.index_save_path)
        if not index_path.exists():
            logger.info("Index path does not exist: %s", self.index_save_path)
            return None
        manifest_path = index_path / self.MANIFEST_FILE
        try:
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            logger.info("Index manifest is missing or invalid; rebuilding index")
            return None
        if saved_manifest != self.expected_manifest:
            logger.info("Index manifest changed; rebuilding index")
            return None
        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded vector index from %s", self.index_save_path)
            return self.vectorstore
        except Exception as exc:
            logger.warning("Failed to load vector index: %s", exc)
            return None
