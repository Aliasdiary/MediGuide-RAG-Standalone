# Third-party data notice

The included `data/medquad_5000.jsonl` is a deterministic subset derived from:

- Dataset: MedQuAD
- Repository: https://github.com/abachaa/MedQuAD
- Authors: Asma Ben Abacha and Dina Demner-Fushman
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Reference: A Question-Entailment Approach to Question Answering,
  BMC Bioinformatics, 2019

The MedQuAD authors removed answers from the A.D.A.M., MedlinePlus Drugs, and
MedlinePlus Herbs and Supplements subsets to respect MedlinePlus copyright.
Those subsets are excluded from this standalone project's ETL and bundled data.

The source organization and original source URL are retained in every JSONL
record so generated answers can attribute and link back to the source material.
