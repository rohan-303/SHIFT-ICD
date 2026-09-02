from __future__ import annotations

from scripts.external_step8.run_step8 import MODEL_REVISION, model_download_request


def test_model_download_request_is_pinned_and_local() -> None:
    request = model_download_request("C:/tmp/model")
    assert request["repo_id"] == "ncbi/MedCPT-Cross-Encoder"
    assert request["revision"] == MODEL_REVISION
    assert request["local_dir"] == "C:/tmp/model"
    assert request["local_dir_use_symlinks"] is False
