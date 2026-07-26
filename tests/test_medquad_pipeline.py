"""Focused tests for MedQuAD preparation and loading."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from rag_modules.data_preparation import DataPreparationModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule
from langchain_core.documents import Document
from finetune.build_sft_dataset import filter_records, medquad_to_sample, write_dataset_info


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_medquad.py"
SPEC = importlib.util.spec_from_file_location("prepare_medquad", SCRIPT_PATH)
prepare_medquad = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_medquad)


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="1" source="NIH" url="https://example.org/topic">
  <Focus>Example condition</Focus>
  <FocusAnnotations><UMLS><CUIs><CUI>C0000001</CUI></CUIs>
  <SemanticTypes><SemanticType>T047</SemanticType></SemanticTypes></UMLS></FocusAnnotations>
  <QAPairs>
    <QAPair pid="1"><Question qid="q1" qtype="symptoms">What are the symptoms?</Question>
    <Answer>Example symptoms.</Answer></QAPair>
    <QAPair pid="2"><Question qid="q2" qtype="treatment">How is it treated?</Question>
    <Answer>Example treatment.</Answer></QAPair>
  </QAPairs>
</Document>
"""


class MedQuADPipelineTest(unittest.TestCase):
    def test_parse_xml_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_dir = Path(temp_dir) / "1_CancerGov_QA"
            xml_dir.mkdir()
            xml_path = xml_dir / "sample.xml"
            xml_path.write_text(SAMPLE_XML, encoding="utf-8")
            records = prepare_medquad.parse_xml(xml_path)

        self.assertEqual(2, len(records))
        self.assertEqual("NIH", records[0]["source_org"])
        self.assertEqual("https://example.org/topic", records[0]["source_url"])
        self.assertEqual("C0000001", records[0]["umls_cui"])
        self.assertEqual("CC BY 4.0", records[0]["license"])

    def test_sampling_is_deterministic(self):
        records = [
            {
                "id": str(index),
                "source_org": f"source-{index % 3}",
                "question_type": f"type-{index % 2}",
            }
            for index in range(30)
        ]
        first = prepare_medquad.stratified_sample(records, 12, 42)
        second = prepare_medquad.stratified_sample(records, 12, 42)
        self.assertEqual(first, second)
        self.assertEqual(12, len(first))

    def test_loader_creates_question_child_and_full_parent(self):
        record = {
            "id": "q1",
            "question": "What are the symptoms?",
            "answer": "Example symptoms.",
            "question_type": "symptoms",
            "focus": "Example condition",
            "source_org": "NIH",
            "source_url": "https://example.org/topic",
            "umls_cui": "C0000001",
            "semantic_type": "T047",
            "source_subset": "1_CancerGov_QA",
            "license": "CC BY 4.0",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "medquad_test.jsonl"
            data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
            module = DataPreparationModule(temp_dir)
            parents = module.load_documents()
            children = module.chunk_documents()
            restored = module.get_parent_documents(children)

        self.assertEqual("What are the symptoms?", children[0].page_content)
        self.assertIn("Example symptoms.", parents[0].page_content)
        self.assertEqual(parents, restored)

    def test_query_synonym_expansion(self):
        expanded = GenerationIntegrationModule._expand_medical_synonyms(
            "hypertension common risk factors"
        )
        self.assertIn("high blood pressure", expanded)

    def test_exact_focus_match_boosts_rrf_result(self):
        module = RetrievalOptimizationModule.__new__(RetrievalOptimizationModule)
        exact = Document(
            page_content="Who is at risk for High Blood Pressure?",
            metadata={"chunk_id": "exact", "focus": "High Blood Pressure"},
        )
        competing = Document(
            page_content="Who is at risk for Hypotension?",
            metadata={"chunk_id": "competing", "focus": "Hypotension"},
        )
        ranked = module._rrf_rerank(
            [competing, exact],
            [competing, exact],
            query="hypertension high blood pressure risk factors",
        )
        self.assertEqual("exact", ranked[0].metadata["chunk_id"])

    def test_sft_sample_preserves_source_and_safety_boundary(self):
        record = {
            "question": "What is High Blood Pressure?",
            "answer": "High blood pressure is a common condition.",
            "question_type": "information",
            "focus": "High Blood Pressure",
            "source_org": "NHLBI",
            "source_url": "http://example.org/hbp",
        }
        sample = medquad_to_sample(record)
        self.assertIn("MedQuAD source", sample["input"])
        self.assertIn("NHLBI", sample["output"])
        self.assertIn("cannot replace professional medical diagnosis", sample["output"])

    def test_sft_filter_removes_long_answers(self):
        records = [
            {
                "question": "Q1",
                "answer": "short",
                "question_type": "information",
                "focus": "Topic",
                "source_org": "NHLBI",
                "source_url": "http://example.org/1",
            },
            {
                "question": "Q2",
                "answer": "x" * 20,
                "question_type": "information",
                "focus": "Topic",
                "source_org": "NHLBI",
                "source_url": "http://example.org/2",
            },
        ]
        filtered = filter_records(records, max_answer_chars=10)
        self.assertEqual(1, len(filtered))
        self.assertEqual("Q1", filtered[0]["question"])

    def test_write_llamafactory_dataset_info(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_dataset_info(output_dir)
            info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertIn("mediguide_sft_train", info)
            self.assertEqual("output", info["mediguide_sft_train"]["columns"]["response"])


if __name__ == "__main__":
    unittest.main()
