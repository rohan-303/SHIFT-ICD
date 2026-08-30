import numpy as np
import torch

from shift_icd.dense.adapters import DenseEncoderSpec, cls_pool, mean_pool


def test_cls_pool_uses_final_layer_first_token():
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    np.testing.assert_allclose(cls_pool(hidden).numpy(), [[0, 1, 2], [6, 7, 8]])


def test_mean_pool_respects_attention_mask():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    np.testing.assert_allclose(mean_pool(hidden, mask).numpy(), [[2.0, 3.0]])


def test_dense_encoder_spec_is_explicit():
    spec = DenseEncoderSpec("id", "sha", "transformers_cls", 64, 8)
    assert spec.model_id == "id"
    assert spec.revision == "sha"
    assert spec.pooling == "transformers_cls"
