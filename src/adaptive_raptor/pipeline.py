"""The QA pipeline shared by every experiment runner.

    dataset -> chunker -> RAPTOR index (cached) -> policy -> retrieval -> generation
            -> answer F1 -> execution traces

Swapping the chunker (PIC), the policy (router) or the generator is a config change; this
module does not need to be edited. Placeholder components are rejected *before* any index
is built or any API is called (see adaptive_raptor.stub).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_raptor.chunking import build_chunker
from adaptive_raptor.config.schema import ChunkingVariant, ConfigRef, ExperimentConfig
from adaptive_raptor.data import DatasetSplit, load_dataset
from adaptive_raptor.evaluation.metrics import answer_f1
from adaptive_raptor.evaluation.traces import ExecutionTrace, TraceWriter
from adaptive_raptor.generation import build_generator
from adaptive_raptor.policy import CatalogPolicy, ConfigPolicy, PlannedConfig
from adaptive_raptor.stub import ensure_implemented

log = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    n_documents: int = 0
    n_questions: int = 0
    n_traces: int = 0
    n_errors: int = 0
    indexes_built: int = 0
    indexes_reused: int = 0
    index_errors: int = 0
    summarizer_usage: dict[str, Any] = field(default_factory=dict)
    dataset_manifest: dict[str, Any] = field(default_factory=dict)


class QAPipeline:
    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: Path,
        *,
        policy: ConfigPolicy | None = None,
        variants: list[ChunkingVariant] | None = None,
        dataset: DatasetSplit | None = None,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.policy = policy or CatalogPolicy(config.retrieval)
        self.variants = variants or list(config.chunking.variants)
        self.chunkers = {v.name: build_chunker(v) for v in self.variants}
        # Fail fast on placeholders, before data loading, index builds or API calls.
        ensure_implemented(self.policy, *self.chunkers.values())
        self._dataset = dataset

    # -- component construction (lazy upstream imports) ---------------------------------

    def _components(self, dataset: DatasetSplit):
        from adaptive_raptor.indexing import IndexBuilder
        from adaptive_raptor.models.embedding import make_embedding_model
        from adaptive_raptor.models.summarization import make_summarization_model
        from adaptive_raptor.retrieval import RetrievalExecutor

        cfg, rt = self.config, self.config.runtime
        embedding = make_embedding_model(cfg.raptor.embedding)
        builder = IndexBuilder(
            cfg.raptor,
            dataset=dataset.name,
            dataset_version=str(dataset.manifest.get("version") or cfg.dataset.version),
            embedding_model=embedding,
            summarizer_factory=lambda: make_summarization_model(
                cfg.raptor.summarization,
                cache_dir=rt.cache_dir / "summaries",
                use_cache=rt.use_cache,
                max_retries=rt.max_retries,
            ),
        )
        executor = RetrievalExecutor(cfg.retrieval, embedding, cfg.raptor.tokenizer)
        generator = build_generator(
            cfg.generation,
            cache_dir=rt.cache_dir / "generations",
            use_cache=rt.use_cache,
            max_retries=rt.max_retries,
        )
        return builder, executor, generator

    # -- main loop ------------------------------------------------------------------------

    def run(self) -> PipelineReport:
        cfg = self.config
        dataset = self._dataset or load_dataset(cfg.dataset)
        report = PipelineReport(
            n_documents=len(dataset.documents),
            n_questions=len(dataset.questions),
            dataset_manifest=dataset.manifest,
        )
        builder, executor, generator = self._components(dataset)
        by_doc = dataset.questions_by_doc()
        traces = TraceWriter(self.run_dir / "traces.jsonl")
        index_log = TraceWriter(self.run_dir / "indexes.jsonl")
        pool = ThreadPoolExecutor(max_workers=cfg.runtime.concurrency)
        try:
            for variant in self.variants:
                chunker = self.chunkers[variant.name]
                for seed in cfg.raptor.seeds:
                    for doc in dataset.documents:
                        questions = by_doc.get(doc.doc_id, [])
                        if not questions:
                            continue
                        handle, err = self._index(builder, chunker, variant, doc, seed)
                        index_log.write(_IndexRecord(doc.doc_id, variant.name, seed, handle, err))
                        if handle is None:
                            report.index_errors += 1
                        elif handle.cache_hit:
                            report.indexes_reused += 1
                        else:
                            report.indexes_built += 1
                        for t in self._answer_doc(
                            pool, executor, generator, handle, err, variant, seed, questions
                        ):
                            traces.write(t)
                            report.n_traces += 1
                            report.n_errors += t.error is not None
        finally:
            pool.shutdown(wait=True)
            traces.close()
            index_log.close()
        usage = getattr(builder._summarizer, "usage", None)
        report.summarizer_usage = dict(usage) if usage else {}
        return report

    def _index(self, builder, chunker, variant, doc, seed):
        from adaptive_raptor.models.credentials import MissingCredentialsError

        try:
            chunks = chunker.chunk(doc)
            return builder.load_or_build(doc.doc_id, variant, chunks, seed), None
        except (MissingCredentialsError, NotImplementedError):
            raise  # configuration problems abort the run
        except Exception as exc:  # one bad document must not kill the run
            log.exception("index failed for %s", doc.doc_id)
            return None, f"{type(exc).__name__}: {exc}"

    def _answer_doc(self, pool, executor, generator, handle, index_error, variant, seed, questions):
        cfg = self.config
        jobs = []
        for q in questions:
            base = dict(
                question_id=q.question_id,
                doc_id=q.doc_id,
                index_variant=variant.name,
                seed=seed,
                run_label=cfg.experiment.run_label,
                question=q.text,
                index_key_hash=handle.key_hash if handle else "",
                index_cache_hit=bool(handle and handle.cache_hit),
            )
            if handle is None:
                jobs.append(
                    _done(ExecutionTrace(config_id="*", error=f"index: {index_error}", **base))
                )
                continue
            embedding = executor.embed(q.text)  # cached; reused by every config
            for planned in self.policy.configs_for(q, embedding):
                jobs.append(pool.submit(self._one, executor, generator, handle, q, planned, base))
        return [j.result() for j in jobs]  # submission order -> deterministic trace order

    def _one(self, executor, generator, handle, q, planned: PlannedConfig, base) -> ExecutionTrace:
        cfg_id = planned.id if isinstance(planned, ConfigRef) else planned
        trace = ExecutionTrace(config_id=cfg_id, **base)
        try:
            if isinstance(planned, ConfigRef):
                r = executor.retrieve(q.text, handle, planned)
            else:
                r = executor.retrieve_original(q.text, handle)
            trace.node_ids, trace.levels = r.node_ids, r.levels
            trace.context_tokens, trace.retrieval_ms = r.tokens_used, round(r.latency_ms, 3)
            if self.config.evaluation.save_contexts:
                trace.contexts = r.texts
            g = generator.generate(q.text, r.texts)
            trace.answer, trace.model_version = g.text, g.model_version
            trace.input_tokens, trace.output_tokens = g.input_tokens, g.output_tokens
            trace.generation_ms, trace.generation_cache_hit = round(g.latency_ms, 1), g.cache_hit
            trace.answer_f1, trace.answer_type = answer_f1(g.text, q.answers)
        except Exception as exc:
            from adaptive_raptor.models.credentials import MissingCredentialsError

            if isinstance(exc, MissingCredentialsError):
                raise
            trace.error = f"{type(exc).__name__}: {exc}"
        return trace


class _done:
    """A finished value with the Future.result() interface."""

    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value


class _IndexRecord:
    def __init__(self, doc_id, variant, seed, handle, error) -> None:
        self.d = {
            "doc_id": doc_id,
            "variant": variant,
            "seed": seed,
            "error": error,
            "cache_hit": handle.cache_hit if handle else None,
            "key_hash": handle.key_hash if handle else None,
            "path": str(handle.path) if handle else None,
            "build_stats": handle.build_stats if handle else None,
        }

    def to_dict(self) -> dict:
        return self.d
