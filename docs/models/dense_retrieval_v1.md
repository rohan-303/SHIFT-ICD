# Dense Retrieval Model Provenance v1

This document records the declared zero-shot models for `dense_v1`. Hub revisions are immutable commit SHAs obtained from the Hugging Face Hub API before evaluation. Model weights are not stored in this repository.

## SapBERT

- Name: SapBERT
- Hub ID: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- Revision: `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`
- Architecture: BERT encoder (`BertModel`), 12 layers, 12 heads
- Parameters: approximately 110M (BERT-base scale; exact count recorded after loading)
- Dimension: 768
- Tokenizer: repository-provided BERT tokenizer, vocabulary 30,522
- Maximum sequence length: 512 positions; experiment cap is selected by the preregistered 99.9% coverage rule
- Pooling: final-layer `[CLS]`, following the official model card
- Similarity: L2-normalized cosine / dot product
- License: Apache-2.0 according to Hub card metadata
- Publication: Liu et al., “Self-Alignment Pretraining for Biomedical Entity Representations,” NAACL-HLT 2021; arXiv:2010.11784
- Objective/resources: self-alignment pretraining using UMLS 2020AA English concepts, initialized from PubMedBERT full-text
- Official sources: [model card](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext), [paper](https://arxiv.org/abs/2010.11784)

## BioLORD-2023

- Name: BioLORD-2023
- Hub ID: `FremyCompany/BioLORD-2023`
- Revision: `167aab527b238a50ca65224e6319215d2ff4fc9f`
- Architecture: MPNet encoder (`MPNetModel`), 12 layers, 12 heads
- Parameters: approximately 109M (MPNet-base scale; exact count recorded after loading)
- Dimension: 768
- Tokenizer: repository-provided MPNet tokenizer, vocabulary 30,527
- Maximum sequence length: 514 positions; experiment cap is selected by the preregistered 99.9% coverage rule
- Pooling: native Sentence-Transformers pipeline/configuration; no manual replacement of configured pooling
- Similarity: normalized Sentence-Transformers embedding cosine similarity
- License: Hub card declares `other`, with MIT for the author’s contributions and UMLS/SNOMED CT licensing obligations for training-derived resources
- Publication: Remy, Demuynck, and Demeester, “BioLORD-2023: semantic textual representations fusing large language models and clinical knowledge graph insights,” JAMIA 2024; arXiv:2311.16075
- Objective/resources: biomedical concept and clinical-sentence representation grounded in definitions, biomedical ontologies, BioLORD-Dataset, and AGCT-Dataset; based on all-mpnet-base-v2
- Restrictions: local research evaluation may proceed under the model card terms; weights and derived redistribution are excluded from this repository; users must independently satisfy UMLS/SNOMED CT licensing requirements
- Official sources: [model card](https://huggingface.co/FremyCompany/BioLORD-2023), [paper](https://arxiv.org/abs/2311.16075)

## MedCPT Query Encoder

- Name: MedCPT Query Encoder
- Hub ID: `ncbi/MedCPT-Query-Encoder`
- Revision: `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`
- Architecture: BERT encoder (`BertModel`), 12 layers, 12 heads
- Parameters: approximately 110M (BERT-base scale; exact count recorded after loading)
- Dimension: 768
- Tokenizer: repository-provided BERT tokenizer, vocabulary 30,522
- Maximum sequence length: 512 positions; experiment cap is selected by the preregistered 99.9% coverage rule
- Pooling: final-layer `[CLS]`, following the official model card
- Similarity: L2-normalized cosine / dot product
- License: Hub card declares `other` / public-domain metadata and points to the repository `LICENSE`; users must review the license shipped with the pinned revision
- Publication: Jin et al., “MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval,” Bioinformatics 2023; arXiv:2307.00589
- Objective/resources: contrastive pretraining on 255M PubMed search query–article pairs
- Primary direction: Query Encoder is used symmetrically for both source and target ICD descriptions. The Article Encoder is not substituted. Any Query→Article experiment would be a separate DEV-only ablation.
- Official sources: [model card](https://huggingface.co/ncbi/MedCPT-Query-Encoder), [paper](https://arxiv.org/abs/2307.00589)

## Qwen3-Embedding-0.6B control

- Name: Qwen3-Embedding-0.6B
- Hub ID: `Qwen/Qwen3-Embedding-0.6B`
- Revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- Architecture: Qwen3 causal-transformer embedding model (`Qwen3ForCausalLM`), 28 layers, 16 heads
- Parameters: approximately 0.6B
- Dimension: 1024 (the model supports configurable dimensions; this experiment uses the native 1024 dimension)
- Tokenizer: repository-provided Qwen3 tokenizer, vocabulary 151,669
- Maximum sequence length: 32,768 positions; experiment cap is selected by the preregistered 99.9% coverage rule
- Pooling: official last-token pooling with left-padding handling
- Similarity: L2-normalized cosine / dot product
- License: Apache-2.0 according to Hub card metadata
- Publication: Zhang et al., “Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models,” arXiv:2506.05176
- Objective/resources: general multilingual embedding and retrieval training; this is a declared control, not the proposed biomedical model
- Instruction: exactly `Retrieve the ICD diagnosis concept that is semantically equivalent or the closest valid cross-version mapping to the source diagnosis description.` on source queries only; target documents receive no instruction or metadata
- Official sources: [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [paper](https://arxiv.org/abs/2506.05176)

## Download and cache policy

- Download date and exact cache paths are recorded in `artifacts/experiments/dense_v1/models.json` and `environment.json`.
- Weights use the standard Hugging Face cache or an ignored project-local cache.
- No model weights are copied into Git, release archives, or public project artifacts.
- Safetensors files are preferred whenever the pinned repository provides them.
- Exact revision SHA, tokenizer revision, model config, file list, and hashes are recorded after successful download.
