import pytest

from shift_icd.reranking.inference import (
    ChunkManifest,
    ThermalGuard,
    ThermalState,
    deterministic_pair_order,
)


def test_chunk_manifest_resume_and_hash_validation(tmp_path):
    m = ChunkManifest(tmp_path / "manifest.json", expected_pairs=4)
    payload = b"rows-0-2"
    m.add_chunk("0000", 0, 2, payload)
    m.save()
    loaded = ChunkManifest.load(tmp_path / "manifest.json")
    assert loaded.completed_pairs == 2
    assert loaded.validate_chunk("0000", payload)
    assert not loaded.validate_chunk("0000", b"tampered")


def test_duplicate_chunk_rejected(tmp_path):
    m = ChunkManifest(tmp_path / "manifest.json", expected_pairs=4)
    m.add_chunk("0000", 0, 2, b"x")
    with pytest.raises(ValueError):
        m.add_chunk("0000", 0, 2, b"x")


def test_missing_chunk_detection(tmp_path):
    m = ChunkManifest(tmp_path / "manifest.json", expected_pairs=4)
    m.add_chunk("0000", 0, 2, b"x")
    assert m.missing_ranges() == [(2, 4)]


def test_deterministic_pair_order():
    rows = [{"benchmark_id": "b", "retriever_rank": 2}, {"benchmark_id": "a", "retriever_rank": 1}]
    assert deterministic_pair_order(rows) == deterministic_pair_order(rows)
    assert [r["benchmark_id"] for r in deterministic_pair_order(rows)] == ["a", "b"]


def test_thermal_soft_pause_and_resume():
    temps = iter([70.0, 80.0, 72.0])
    guard = ThermalGuard(read_temperature=lambda: next(temps), soft_pause=80, hard_stop=83, resume=72)
    assert guard.check() == ThermalState.OK
    assert guard.check() == ThermalState.COOLDOWN
    assert guard.check() == ThermalState.OK
    assert guard.pause_count == 1


def test_thermal_hard_stop():
    guard = ThermalGuard(read_temperature=lambda: 83.0, soft_pause=80, hard_stop=83, resume=72)
    assert guard.check() == ThermalState.HARD_STOP
    assert guard.hard_stop


def test_resumable_score_persists_and_skips_completed_chunks(tmp_path):
    from shift_icd.reranking.inference import resumable_score

    pairs = [{"benchmark_id": f"b{i}", "retriever_rank": 1} for i in range(4)]
    calls = []

    def score_fn(chunk):
        calls.append(len(chunk))
        return str([row["benchmark_id"] for row in chunk]).encode()

    manifest = ChunkManifest(tmp_path / "manifest.json", expected_pairs=4)
    assert resumable_score(pairs, score_fn, tmp_path / "chunks", manifest, 2) == ThermalState.OK
    assert calls == [2, 2]
    assert resumable_score(pairs, score_fn, tmp_path / "chunks", ChunkManifest.load(tmp_path / "manifest.json"), 2) == ThermalState.OK
    assert calls == [2, 2]
