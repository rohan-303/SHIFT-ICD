# SHIFT-MAP v1.3 retrieval-stage freeze

retriever_name: SHIFT-MAP v1.3 L2 bi-encoder
base: FremyCompany/BioLORD-2023
base_revision: 167aab527b238a50ca65224e6319215d2ff4fc9f
canonical_seed: 17
canonical_epoch: 3
checkpoint_hash: 7a859ff478d801f98a17fc966cb0ae81ba60362725add2735d91c7e01f4fe1d
positive_policy: P2
loss: L2 set-positive InfoNCE
negative_strategy: N1_RANDOM
learning_rate: 2e-5
evaluator_version: 2.0
benchmark_version: 1.0
canonical_schema_version: 1.0
target_terminology: ICD-10-CM 2018
target_count: 17513
target_corpus_hash: a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a
candidate_k: 100
similarity: cosine
embedding_normalization: L2 normalized
checkpoint_release: LOCAL ONLY
test_exposure: historically exposed; fully disclosed
status: FROZEN

Step 8 must consume only the candidate manifests linked to this contract. `candidate_is_gold` is evaluation/training bookkeeping and must never be included in model input text. No clinical-use or clinical-equivalence claim is made.
