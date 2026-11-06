"""QASPER parsing, schema integrity, subsets, and the download pipeline (no network)."""

import hashlib
import io
import json
import tarfile

import pytest

from adaptive_raptor.config.schema import DatasetSection
from adaptive_raptor.data import FIXTURE_PATH, load_dataset, qasper
from adaptive_raptor.data import download as dl
from adaptive_raptor.data.subsets import read_id_file, sample_doc_ids, select
from tests.conftest import REAL_QASPER

RAW = json.loads(FIXTURE_PATH.read_text())


def test_parse_fixture_schema():
    docs, qs, stats = qasper.parse_split(RAW["dev"])
    assert [d.doc_id for d in docs] == ["fixture.0001", "fixture.0002"]
    assert stats["skipped_annotations"] == 0
    for d in docs:
        assert d.text.startswith(d.title)
        for p in d.paragraphs:
            assert d.text[p.start : p.end] == p.text  # exact offsets
        assert d.paragraphs[0].section == "Abstract"
    types = {a.answer_type for q in qs for a in q.answers}
    assert types == {"extractive", "abstractive", "boolean", "none"}
    by_id = {q.question_id: q for q in qs}
    assert by_id["fixture.0001-q3"].answers[0].text == "Unanswerable"
    assert {a.text for a in by_id["fixture.0002-q1"].answers} == {"Yes", "No"}
    assert by_id["fixture.0001-q0"].answers[0].text == "3,044 sentences in 100 dialogs"


def test_records_roundtrip():
    docs, qs, _ = qasper.parse_split(RAW["dev"])
    from adaptive_raptor.data.records import Document, Question

    assert Document.from_dict(docs[0].to_dict()) == docs[0]
    assert Question.from_dict(qs[0].to_dict()) == qs[0]


def test_subsets_are_deterministic_and_paper_level():
    ids = [f"p{i}" for i in range(50)]
    a, b = sample_doc_ids(ids, 10, seed=3), sample_doc_ids(ids, 10, seed=3)
    assert a == b and len(a) == 10 and a != sample_doc_ids(ids, 10, seed=4)
    docs, qs, _ = qasper.parse_split(RAW["dev"])
    d2, q2 = select(docs, qs, exclude=["fixture.0001"])
    assert [d.doc_id for d in d2] == ["fixture.0002"]
    assert all(q.doc_id == "fixture.0002" for q in q2)
    with pytest.raises(ValueError):
        select(docs, qs, include=["nope"])


def test_load_dataset_fixture_with_limits():
    split = load_dataset(DatasetSection(name="qasper_tiny", split="dev", max_documents=1))
    assert len(split.documents) == 1
    assert {q.doc_id for q in split.questions} == {split.documents[0].doc_id}


def test_load_dataset_missing_processed_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_datasets"):
        load_dataset(DatasetSection(name="qasper", path=tmp_path))


def _fake_release(tmp_path):
    """Build tarballs shaped like the official release from the fixture."""
    files = []
    layout = [
        (
            "train-dev.tgz",
            {
                "qasper-train-v0.3.json": RAW["train"],
                "qasper-dev-v0.3.json": RAW["dev"],
                "README.md": "readme",
            },
        ),
        (
            "test.tgz",
            {
                "qasper-test-v0.3.json": RAW["test"],
                "qasper_evaluator.py": "# evaluator",
                "README-test.md": "readme",
            },
        ),
    ]
    for name, members in layout:
        path = tmp_path / "src" / name
        path.parent.mkdir(exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            for mname, content in members.items():
                data = (content if isinstance(content, str) else json.dumps(content)).encode()
                info = tarfile.TarInfo(mname)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        files.append(
            {
                "url": path.as_uri(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "members": list(members),
            }
        )
    return {
        "version": "v0.3",
        "homepage": "x",
        "license": "x",
        "files": files,
        "expected": {
            "train": {"documents": 0, "questions": 0},
            "dev": {"documents": 2, "questions": 6},
            "test": {"documents": 1, "questions": 1},
        },
    }


def test_download_preprocess_verify_idempotent(tmp_path, monkeypatch):
    monkeypatch.setitem(dl.SOURCES, "qasper", _fake_release(tmp_path))
    root = tmp_path / "data"
    assert dl.main(["--dataset", "qasper", "--data-dir", str(root)]) == 0
    manifest = json.loads((root / "processed/qasper/v0.3/manifest.json").read_text())
    assert manifest["splits"]["dev"]["questions"] == 6
    assert manifest["preprocess_version"] == qasper.PREPROCESS_VERSION

    def no_network(*a, **k):
        raise AssertionError("second run must not download")

    monkeypatch.setattr(dl, "_download", no_network)
    assert dl.main(["--dataset", "qasper", "--data-dir", str(root)]) == 0
    assert dl.main(["--dataset", "qasper", "--data-dir", str(root), "--verify"]) == 0

    split = load_dataset(DatasetSection(name="qasper", path=root, split="dev"))
    assert len(split.questions) == 6 and split.manifest["version"] == "v0.3"

    (root / "processed/qasper/v0.3/dev/questions.jsonl").write_text("")  # tamper
    assert dl.main(["--dataset", "qasper", "--data-dir", str(root), "--verify"]) == 1


def test_download_rejects_bad_checksum(tmp_path, monkeypatch):
    src = _fake_release(tmp_path)
    src["files"][0]["sha256"] = "0" * 64
    monkeypatch.setitem(dl.SOURCES, "qasper", src)
    assert dl.main(["--dataset", "qasper", "--data-dir", str(tmp_path / "d")]) == 1


def test_make_subset(tmp_path, monkeypatch):
    monkeypatch.setitem(dl.SOURCES, "qasper", _fake_release(tmp_path))
    root = tmp_path / "data"
    dl.main(["--dataset", "qasper", "--data-dir", str(root)])
    args = [
        "--dataset",
        "qasper",
        "--data-dir",
        str(root),
        "--make-subset",
        "pilot",
        "--split",
        "dev",
        "--n-docs",
        "1",
        "--seed",
        "0",
    ]
    assert dl.main(args) == 0
    ids = read_id_file(root / "subsets/qasper/v0.3/dev_pilot.txt")
    assert len(ids) == 1


@pytest.mark.skipif(not REAL_QASPER.is_dir(), reason="QASPER not downloaded")
def test_real_qasper_integrity():
    """Runs only after ./scripts/download_datasets.sh --dataset qasper (offline)."""
    assert dl.verify_qasper() == []
    docs, qs = qasper.load_processed(REAL_QASPER, "dev")
    assert len(docs) == 281 and len(qs) == 1005
    assert all(q.answers for q in qs)
