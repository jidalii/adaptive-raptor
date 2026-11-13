"""Download, verify and preprocess datasets. The ONLY code path that touches the network
for data. Idempotent: verified files are never downloaded or reprocessed twice.

    python -m adaptive_raptor.data.download --dataset qasper
    python -m adaptive_raptor.data.download --dataset qasper --verify
    python -m adaptive_raptor.data.download --dataset qasper --make-subset pilot \\
        --split dev --n-docs 30 --seed 0

Layout (under the data root, see adaptive_raptor.paths):
    raw/qasper/v0.3/<archives>, raw/qasper/v0.3/extracted/<json + evaluator>
    processed/qasper/v0.3/{train,dev,test}/{documents,questions}.jsonl, manifest.json
    subsets/qasper/v0.3/<split>_<name>.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from adaptive_raptor.data import narrativeqa, qasper
from adaptive_raptor.data.subsets import read_id_file, sample_doc_ids, write_id_file
from adaptive_raptor.paths import data_root, processed_dir, raw_dir

QASPER_BASE = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
SOURCES = {
    "qasper": {
        "version": "v0.3",
        "homepage": "https://allenai.org/data/qasper",
        "license": "see homepage (not stated in the archive README)",
        "files": [
            {
                "url": f"{QASPER_BASE}/qasper-train-dev-v0.3.tgz",
                "sha256": "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",
                "members": ["qasper-train-v0.3.json", "qasper-dev-v0.3.json", "README.md"],
            },
            {
                "url": f"{QASPER_BASE}/qasper-test-and-evaluator-v0.3.tgz",
                "sha256": "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72",
                "members": ["qasper-test-v0.3.json", "qasper_evaluator.py", "README-test.md"],
            },
        ],
        # Official release statistics; --verify checks these exactly.
        "expected": {
            "train": {"documents": 888, "questions": 2593},
            "dev": {"documents": 281, "questions": 1005},
            "test": {"documents": 416, "questions": 1451},
        },
    }
}


NQA_REVISION = "2e643e7363944af1c33a652d1c87320d0871c4e4"
NQA_VERSION = NQA_REVISION[:8]
NQA_BASE = f"https://huggingface.co/datasets/deepmind/narrativeqa/resolve/{NQA_REVISION}/data"
# Only the splits we use are listed. The test split is not downloaded (final evaluation only).
NQA_SOURCES = {
    "version": NQA_VERSION,
    "revision": NQA_REVISION,
    "homepage": "https://huggingface.co/datasets/deepmind/narrativeqa",
    "license": "apache-2.0 (dataset card)",
    "splits": {
        "dev": [
            (
                "validation-00000-of-00003.parquet",
                "5bd4cfb854ec4c22b98583fa4b6b16f5e6f7461d8bfc13cfe8e7eb12b30871d0",
            ),
            (
                "validation-00001-of-00003.parquet",
                "2f0728b3f819c1ccf67a1d0e1c37bfe8523063a46a5742275770d3db2c2ed799",
            ),
            (
                "validation-00002-of-00003.parquet",
                "3f6da16ec6688f788947023d84408bb7af71de7fd379d86b4ea517c42297d39b",
            ),
        ],
    },
    "expected": {"dev": {"documents": 115, "questions": 3461}},
}
NQA_CONVERT = (
    "uv run --isolated --no-project --with pyarrow python "
    "scripts/narrativeqa_parquet_to_jsonl.py --split {hf} --raw-dir {raw}"
)


class DatasetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:
        shutil.copyfileobj(resp, out)
    got = sha256_file(part)
    if got != expected_sha:
        part.unlink(missing_ok=True)
        raise DatasetError(f"checksum mismatch for {url}: expected {expected_sha}, got {got}")
    part.replace(dest)


def _extract(archive: Path, members: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        names = {m.name.lstrip("./"): m for m in tar.getmembers()}
        for name in members:
            member = names.get(name)
            if member is None or not member.isfile():
                raise DatasetError(f"{archive.name} has no regular file '{name}'")
            src = tar.extractfile(member)
            assert src is not None
            with src, (out_dir / Path(name).name).open("wb") as dst:  # flat, no path traversal
                shutil.copyfileobj(src, dst)


def fetch_qasper(root: Path | None = None, force: bool = False) -> Path:
    src = SOURCES["qasper"]
    version = src["version"]
    rdir = raw_dir("qasper", version, root)
    extracted = rdir / "extracted"
    for f in src["files"]:
        archive = rdir / Path(f["url"]).name
        if force or not archive.is_file() or sha256_file(archive) != f["sha256"]:
            _download(f["url"], archive, f["sha256"])
        else:
            print(f"  cached  {archive}")
        if force or not all((extracted / m).is_file() for m in f["members"]):
            _extract(archive, f["members"], extracted)
    return extracted


def preprocess_qasper(root: Path | None = None, force: bool = False) -> Path:
    src = SOURCES["qasper"]
    version = src["version"]
    extracted = raw_dir("qasper", version, root) / "extracted"
    pdir = processed_dir("qasper", version, root)
    manifest_path = pdir / "manifest.json"
    raw_hashes = {f["url"]: f["sha256"] for f in src["files"]}

    if manifest_path.is_file() and not force:
        m = json.loads(manifest_path.read_text())
        if (
            m.get("preprocess_version") == qasper.PREPROCESS_VERSION
            and m.get("raw_sha256") == raw_hashes
            and _processed_files_ok(pdir, m)
        ):
            print(f"  cached  {pdir} (preprocess v{qasper.PREPROCESS_VERSION})")
            return pdir

    splits = {}
    for split in qasper.SPLITS:
        raw_path = extracted / qasper.raw_filename(split, version)
        raw = json.loads(raw_path.read_text())
        docs, qs, stats = qasper.parse_split(raw)
        dpath, qpath = qasper.processed_split_paths(pdir, split)
        qasper.write_jsonl(dpath, (d.to_dict() for d in docs))
        qasper.write_jsonl(qpath, (q.to_dict() for q in qs))
        splits[split] = {
            "documents": len(docs),
            "questions": len(qs),
            **stats,
            "files": {
                "documents.jsonl": sha256_file(dpath),
                "questions.jsonl": sha256_file(qpath),
            },
        }
        print(f"  {split:5s}: {len(docs)} papers, {len(qs)} questions")

    manifest = {
        "dataset": "qasper",
        "version": version,
        "homepage": src["homepage"],
        "license": src["license"],
        "sources": [f["url"] for f in src["files"]],
        "raw_sha256": raw_hashes,
        "preprocess_version": qasper.PREPROCESS_VERSION,
        "preprocessing": {
            "document_text": "title + abstract + non-empty section paragraphs, '\\n\\n'-joined",
            "answers": "official qasper_evaluator.py extraction (extractive/abstractive/"
            "boolean/none)",
            "split_names": {"dev": "qasper-dev-v0.3.json"},
        },
        "created_at": datetime.now(UTC).isoformat(),
        "splits": splits,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return pdir


def _processed_files_ok(pdir: Path, manifest: dict) -> bool:
    for split, info in manifest.get("splits", {}).items():
        for fname, sha in info.get("files", {}).items():
            p = pdir / split / fname
            if not p.is_file() or sha256_file(p) != sha:
                return False
    return True


def verify_qasper(root: Path | None = None) -> list[str]:
    """Offline integrity check. Returns a list of problems (empty = OK)."""
    src = SOURCES["qasper"]
    version = src["version"]
    problems: list[str] = []
    rdir = raw_dir("qasper", version, root)
    for f in src["files"]:
        archive = rdir / Path(f["url"]).name
        if not archive.is_file():
            problems.append(f"missing archive {archive}")
        elif sha256_file(archive) != f["sha256"]:
            problems.append(f"checksum mismatch: {archive}")
    pdir = processed_dir("qasper", version, root)
    mpath = pdir / "manifest.json"
    if not mpath.is_file():
        problems.append(f"missing {mpath} (run the download without --verify)")
        return problems
    manifest = json.loads(mpath.read_text())
    if not _processed_files_ok(pdir, manifest):
        problems.append("processed files do not match manifest checksums")
    for split, exp in src["expected"].items():
        try:
            docs, qs = qasper.load_processed(pdir, split)
        except FileNotFoundError as exc:
            problems.append(str(exc))
            continue
        if len(docs) != exp["documents"] or len(qs) != exp["questions"]:
            problems.append(
                f"{split}: {len(docs)} papers / {len(qs)} questions, expected "
                f"{exp['documents']} / {exp['questions']}"
            )
        doc_ids = {d.doc_id for d in docs}
        if any(q.doc_id not in doc_ids for q in qs):
            problems.append(f"{split}: questions reference unknown papers")
        if len({q.question_id for q in qs}) != len(qs):
            problems.append(f"{split}: duplicate question ids")
        for d in docs:
            if any(d.text[p.start : p.end] != p.text for p in d.paragraphs):
                problems.append(f"{split}: paragraph offsets wrong in {d.doc_id}")
                break
    ids = {}
    for split in src["expected"]:
        try:
            ids[split] = {d.doc_id for d in qasper.load_processed(pdir, split)[0]}
        except FileNotFoundError:
            pass
    for a in ids:
        for b in ids:
            if a < b and ids[a] & ids[b]:
                problems.append(f"papers shared between {a} and {b}")
    return problems


def make_subset(
    name: str,
    split: str,
    n_docs: int,
    seed: int,
    root: Path | None = None,
    exclude_file: Path | None = None,
) -> Path:
    version = SOURCES["qasper"]["version"]
    docs, _ = qasper.load_processed(processed_dir("qasper", version, root), split)
    ids = [d.doc_id for d in docs]
    if exclude_file is not None:
        drop = set(read_id_file(exclude_file))
        ids = [i for i in ids if i not in drop]
    chosen = sample_doc_ids(ids, n_docs, seed)
    out = data_root(root) / "subsets" / "qasper" / version / f"{split}_{name}.txt"
    header = f"qasper {version} {split} subset '{name}': n_docs={n_docs} seed={seed}"
    if exclude_file:
        header += f" exclude={exclude_file}"
    write_id_file(out, chosen, header)
    return out


def fetch_narrativeqa(split: str, root: Path | None = None, force: bool = False) -> Path:
    rdir = raw_dir("narrativeqa", NQA_VERSION, root)
    for name, sha in NQA_SOURCES["splits"][split]:
        dest = rdir / name
        if force or not dest.is_file() or sha256_file(dest) != sha:
            _download(f"{NQA_BASE}/{name}", dest, sha)
        else:
            print(f"  cached  {dest}")
    return rdir


def preprocess_narrativeqa(split: str, root: Path | None = None) -> Path:
    """Raw JSONL (from the parquet converter) -> processed records + meta.jsonl."""
    rdir = raw_dir("narrativeqa", NQA_VERSION, root)
    hf = narrativeqa.SPLITS[split]
    try:
        raw_docs, raw_qs = narrativeqa.read_raw(rdir, hf)
    except FileNotFoundError as exc:
        raise DatasetError(
            f"{exc} is missing. Convert the parquet shards first:\n  "
            + NQA_CONVERT.format(hf=hf, raw=rdir)
        ) from exc
    docs, qs, metas, stats = narrativeqa.parse_split(raw_docs, raw_qs)
    exp = NQA_SOURCES["expected"].get(split)
    if exp and (len(docs), len(qs)) != (exp["documents"], exp["questions"]):
        raise DatasetError(f"{split}: {len(docs)} docs / {len(qs)} questions, expected {exp}")
    pdir = processed_dir("narrativeqa", NQA_VERSION, root)
    dpath, qpath = qasper.processed_split_paths(pdir, split)
    qasper.write_jsonl(dpath, (d.to_dict() for d in docs))
    qasper.write_jsonl(qpath, (q.to_dict() for q in qs))
    qasper.write_jsonl(pdir / split / "meta.jsonl", metas)
    mpath = pdir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    manifest.update(
        {
            "dataset": "narrativeqa",
            "version": NQA_VERSION,
            "revision": NQA_REVISION,
            "homepage": NQA_SOURCES["homepage"],
            "license": NQA_SOURCES["license"],
            "preprocess_version": narrativeqa.PREPROCESS_VERSION,
            "preprocessing": "see adaptive_raptor/data/narrativeqa.py docstring",
        }
    )
    manifest.setdefault("raw_sha256", {})
    for name, sha in NQA_SOURCES["splits"][split]:
        manifest["raw_sha256"][f"{NQA_BASE}/{name}"] = sha
    manifest.setdefault("sources", [])
    manifest["sources"] = sorted(set(manifest["sources"]) | set(manifest["raw_sha256"]))
    manifest.setdefault("splits", {})[split] = {
        "documents": len(docs),
        "questions": len(qs),
        **stats,
        "created_at": datetime.now(UTC).isoformat(),
        "files": {
            "documents.jsonl": sha256_file(dpath),
            "questions.jsonl": sha256_file(qpath),
        },
    }
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {split}: {len(docs)} documents, {len(qs)} questions, {stats}")
    return pdir


def make_subset_narrativeqa(
    name: str, split: str, n_docs: int, seed: int, root: Path | None = None
) -> Path:
    """Seeded subset stratified by kind (books / movie scripts): n_docs // 2 of each."""
    pdir = processed_dir("narrativeqa", NQA_VERSION, root)
    metas = narrativeqa.load_meta(pdir, split)
    chosen: list[str] = []
    kinds = sorted({m["kind"] for m in metas})
    for i, kind in enumerate(kinds):
        n = n_docs // len(kinds) + (1 if i < n_docs % len(kinds) else 0)
        chosen += sample_doc_ids([m["doc_id"] for m in metas if m["kind"] == kind], n, seed)
    out = data_root(root) / "subsets" / "narrativeqa" / NQA_VERSION / f"{split}_{name}.txt"
    header = (
        f"narrativeqa {NQA_VERSION} {split} subset '{name}': n_docs={n_docs} seed={seed}, "
        f"stratified by kind {kinds}"
    )
    write_id_file(out, sorted(chosen), header)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="adaptive_raptor.data.download", description=__doc__)
    p.add_argument("--dataset", required=True, choices=["qasper", "quality", "narrativeqa"])
    p.add_argument("--data-dir", type=Path, default=None, help="data root (default: ./data)")
    p.add_argument("--verify", action="store_true", help="offline integrity check only")
    p.add_argument("--force", action="store_true", help="re-download and re-process")
    p.add_argument("--make-subset", metavar="NAME", help="write a seeded paper-id subset file")
    p.add_argument("--split", choices=list(qasper.SPLITS), default="dev")
    p.add_argument("--n-docs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", type=Path, default=None, help="id file to exclude from sampling")
    args = p.parse_args(argv)

    if args.dataset == "quality":
        print("QuALITY is not supported yet (backlog T16).", file=sys.stderr)
        return 2
    root = args.data_dir
    if args.dataset == "narrativeqa":
        try:
            if args.split not in NQA_SOURCES["splits"]:
                p.error(f"narrativeqa: only {sorted(NQA_SOURCES['splits'])} are configured")
            if args.make_subset:
                if args.n_docs is None:
                    p.error("--make-subset requires --n-docs")
                out = make_subset_narrativeqa(
                    args.make_subset, args.split, args.n_docs, args.seed, root
                )
                print(f"wrote {out}")
                return 0
            fetch_narrativeqa(args.split, root, force=args.force)
            print(f"done: {preprocess_narrativeqa(args.split, root)}")
            return 0
        except (DatasetError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        if args.verify:
            problems = verify_qasper(root)
            if problems:
                print("QASPER verification FAILED:", file=sys.stderr)
                for prob in problems:
                    print(f"  - {prob}", file=sys.stderr)
                return 1
            print(f"QASPER {SOURCES['qasper']['version']} verified under {data_root(root)}")
            return 0
        if args.make_subset:
            if args.n_docs is None:
                p.error("--make-subset requires --n-docs")
            out = make_subset(
                args.make_subset, args.split, args.n_docs, args.seed, root, args.exclude
            )
            print(f"wrote {out}")
            return 0
        print(f"QASPER -> {data_root(root)}")
        fetch_qasper(root, force=args.force)
        pdir = preprocess_qasper(root, force=args.force)
        problems = verify_qasper(root)
        if problems:
            for prob in problems:
                print(f"  - {prob}", file=sys.stderr)
            return 1
        print(f"done: {pdir}")
        return 0
    except (DatasetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
