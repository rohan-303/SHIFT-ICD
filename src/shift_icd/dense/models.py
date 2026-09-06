from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from shift_icd.dense.adapters import cls_pool, numpy_embeddings
from shift_icd.dense.retrieval import l2_normalize
from shift_icd.dense.text import clean_dense_text, format_qwen_query

QWEN_INSTRUCTION = (
    "Retrieve the ICD diagnosis concept that is semantically equivalent or the closest valid "
    "cross-version mapping to the source diagnosis description."
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str
    kind: str
    license: str
    embedding_dim: int


@dataclass
class EncodingResult:
    embeddings: np.ndarray
    token_lengths: list[int]
    truncation_count: int
    elapsed_seconds: float


class DenseEncoder:
    spec: ModelSpec

    def encode(self, texts: list[str], max_length: int, batch_size: int, is_query: bool = False) -> EncodingResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


class TransformerCLSEncoder(DenseEncoder):
    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True)
        self.model = AutoModel.from_pretrained(spec.model_id, revision=spec.revision, use_safetensors=True).to(self.device)  # type: ignore[no-untyped-call]
        self.model.eval()

    def encode(self, texts: list[str], max_length: int, batch_size: int, is_query: bool = False) -> EncodingResult:
        import time

        cleaned = [clean_dense_text(text) for text in texts]
        token_lengths = [len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in cleaned]
        truncations = sum(length > max_length for length in token_lengths)
        batches: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(cleaned), batch_size):
                encoded = self.tokenizer(
                    cleaned[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                batches.append(l2_normalize(numpy_embeddings(cls_pool(output.last_hidden_state))))
        embeddings = np.concatenate(batches, axis=0) if batches else np.empty((0, self.spec.embedding_dim), dtype=np.float32)
        return EncodingResult(embeddings, token_lengths, truncations, time.perf_counter() - started)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SentenceTransformerEncoder(DenseEncoder):
    def __init__(self, spec: ModelSpec, device: str, qwen: bool = False) -> None:
        self.spec = spec
        self.qwen = qwen
        self.model = SentenceTransformer(spec.model_id, revision=spec.revision, device=device, trust_remote_code=False)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    def encode(self, texts: list[str], max_length: int, batch_size: int, is_query: bool = False) -> EncodingResult:
        import time

        cleaned = [clean_dense_text(text) for text in texts]
        if self.qwen and is_query:
            cleaned = [format_qwen_query(QWEN_INSTRUCTION, text) for text in cleaned]
        token_lengths = [len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in cleaned]
        truncations = sum(length > max_length for length in token_lengths)
        started = time.perf_counter()
        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=str(self.model.device),
            max_length=max_length,
        ).astype(np.float32, copy=False)
        return EncodingResult(embeddings, token_lengths, truncations, time.perf_counter() - started)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_encoder(spec: ModelSpec, device: str) -> DenseEncoder:
    if spec.kind == "transformers_cls":
        return TransformerCLSEncoder(spec, device)
    if spec.kind == "sentence_transformer":
        return SentenceTransformerEncoder(spec, device)
    if spec.kind == "qwen_sentence_transformer":
        return SentenceTransformerEncoder(spec, device, qwen=True)
    raise ValueError(f"unknown encoder kind: {spec.kind}")
