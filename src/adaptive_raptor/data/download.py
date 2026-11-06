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

from adaptive_raptor.data import qasper
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="adaptive_raptor.data.download", description=__doc__)
    p.add_argument("--dataset", required=True, choices=["qasper", "quality"])
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
            out = make_subset(args.make_subset, args.split, args.n_docs, args.seed, root, args.exclude)
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
