"""
MediGuide-RAG configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class MediGuideConfig:
    """Configuration for the medical guidance RAG agent."""

    data_path: str = str(PROJECT_ROOT / "data")
    index_save_path: str = str(PROJECT_ROOT / "medical_vector_index")

    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "qwen3:latest"
    ollama_base_url: str = "http://localhost:11434"
    medquad_url: str = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
    dataset_limit: int = 5000
    dataset_seed: int = 42

    top_k: int = 4
    temperature: float = 0.1
    max_tokens: int = 2048
    evidence_gate_min_score: int = 2
    evidence_gate_max_docs: int = 3
    reranker_model: str = "BAAI/bge-reranker-base"
    use_cross_encoder_reranker: bool = False
    rag_sft_model_path: str = str(PROJECT_ROOT / "finetune" / "export" / "qwen25-3b-mediguide-sft")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MediGuideConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "ollama_base_url": self.ollama_base_url,
            "medquad_url": self.medquad_url,
            "dataset_limit": self.dataset_limit,
            "dataset_seed": self.dataset_seed,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "evidence_gate_min_score": self.evidence_gate_min_score,
            "evidence_gate_max_docs": self.evidence_gate_max_docs,
            "reranker_model": self.reranker_model,
            "use_cross_encoder_reranker": self.use_cross_encoder_reranker,
            "rag_sft_model_path": self.rag_sft_model_path,
        }


DEFAULT_CONFIG = MediGuideConfig()
