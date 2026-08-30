# BioLORD-2023 license and release notes

**Model:** `FremyCompany/BioLORD-2023`
**Pinned revision:** `167aab527b238a50ca65224e6319215d2ff4fc9f`
**Checked:** 2026-08-30

## Source terms

The Hugging Face model card declares:

- Hub license metadata: `ihtsdo-and-nlm-licences`.
- The author’s own contributions are covered by the MIT license.
- The training data originates from UMLS and SNOMED CT.
- Users must ensure that they have appropriate UMLS and SNOMED CT licensing.
- The model card describes the BioLORD dataset and AGCT dataset as training sources.
- The model card requests citation of the BioLORD-2023 publication.

Sources:

- https://huggingface.co/FremyCompany/BioLORD-2023
- https://huggingface.co/FremyCompany/BioLORD-2023/raw/main/README.md
- https://www.nlm.nih.gov/databases/umls.html

## Practical research-release interpretation

This is a provenance and release assessment, not legal advice.

1. **Local academic fine-tuning:** The model card does not state that local research fine-tuning is prohibited. Local fine-tuning is therefore not blocked by the model card, subject to the researcher’s UMLS/SNOMED CT permissions and institutional review.
2. **Public fine-tuned weights:** Redistribution of derivative weights is **unclear** from the model card alone. The repository’s `other` license metadata and upstream data terms should be reviewed before releasing a checkpoint.
3. **UMLS/SNOMED CT constraints:** The model card expressly places responsibility on the user to maintain appropriate licensing. Any redistribution containing or reconstructing restricted terminology content requires separate review.
4. **Attribution:** Preserve the model repository, exact revision, model-card license metadata, BioLORD-2023 paper citation, and applicable UMLS/SNOMED CT attribution/notice requirements.
5. **Potentially releasable artifacts:** Code, configurations, split-generation code, benchmark identifiers, seeds, and evaluation results can generally be prepared for release, but each release should be checked against the applicable data terms. Do not include restricted source terminology dumps merely for reproducibility.
6. **Checkpoint policy:** Keep fine-tuned BioLORD weights private or institutionally controlled unless the licensing terms have been confirmed to permit public redistribution.
7. **Embedding policy:** Treat large derived embeddings as non-redistributable by default until the upstream and source-data terms are confirmed. Release hashes, schemas, and regeneration instructions instead.

## Decision for SHIFT-MAP

The uncertainty does **not** automatically block local research, but it does block an unconditional public-checkpoint release plan. Step 7 must preserve the base revision and provenance and must not promise public BioLORD-derived weights before a licensing review confirms that release is permitted.
