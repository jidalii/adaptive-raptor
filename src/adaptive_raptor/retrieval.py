"""Apply retrieval strategies and token budgets through upstream RAPTOR.

No retrieval algorithm is duplicated: each strategy supplies a level-restricted tree view
to upstream ``TreeRetriever.retrieve_information_collapse_tree``.  The original baseline
calls upstream ``TreeRetriever.retrieve`` unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from adaptive_raptor.config.schema import ConfigRef, RetrievalSection, Strategy
from adaptive_raptor.indexing.builder import IndexHandle
from adaptive_raptor.indexing.store import EMB_KEY

RAPTOR_ORIGINAL_ID = "raptor_original"


@dataclass
class RetrievalResult:
    config_id: str
    node_ids: list[int] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    context: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


def strategy_layers(strategy: Strategy, tree) -> set[int]:
    all_layers = {int(key) for key in tree.layer_to_nodes}
    if strategy is Strategy.LEAF_ONLY:
        return {0}
    if strategy is Strategy.SUMMARY_ONLY:
        return {layer for layer in all_layers if layer >= 1}
    return all_layers


def _layer_view(tree, layers: set[int]):
    from raptor.tree_structures import Tree

    layer_to_nodes = {
        key: nodes for key, nodes in tree.layer_to_nodes.items() if int(key) in layers
    }
    nodes = {node.index: node for values in layer_to_nodes.values() for node in values}
    return Tree(nodes, tree.root_nodes, tree.leaf_nodes, tree.num_layers, layer_to_nodes)


class RetrievalExecutor:
    def __init__(self, cfg: RetrievalSection, embedding_model, tokenizer_name: str = "cl100k_base"):
        import tiktoken

        self.cfg = cfg
        self.embedding_model = embedding_model
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def embed(self, question: str):
        return self.embedding_model.create_embedding(question)

    def _retriever(self, tree):
        from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig

        config = TreeRetrieverConfig(
            tokenizer=self.tokenizer,
            context_embedding_model=EMB_KEY,
            embedding_model=self.embedding_model,
        )
        return TreeRetriever(config, tree)

    def _result(self, config_id: str, nodes, layer_of, context: str, started: float):
        return RetrievalResult(
            config_id=config_id,
            node_ids=[node.index for node in nodes],
            levels=[layer_of[node.index] for node in nodes],
            texts=[node.text for node in nodes],
            context=context,
            tokens_used=sum(len(self.tokenizer.encode(node.text)) for node in nodes),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def retrieve(self, question: str, index: IndexHandle, config: ConfigRef) -> RetrievalResult:
        from raptor.utils import reverse_mapping

        started = time.perf_counter()
        tree = index.tree
        layer_of = reverse_mapping(tree.layer_to_nodes)
        view = _layer_view(tree, strategy_layers(config.strategy, tree))
        if not view.all_nodes:
            return RetrievalResult(
                config_id=config.id, latency_ms=(time.perf_counter() - started) * 1000
            )
        nodes, context = self._retriever(view).retrieve_information_collapse_tree(
            question, top_k=len(view.all_nodes), max_tokens=config.budget
        )
        return self._result(config.id, nodes, layer_of, context, started)

    def retrieve_original(self, question: str, index: IndexHandle) -> RetrievalResult:
        started = time.perf_counter()
        params = self.cfg.raptor_original
        context, info = self._retriever(index.tree).retrieve(
            question,
            top_k=params.top_k,
            max_tokens=params.max_tokens,
            collapse_tree=params.collapse_tree,
            return_layer_information=True,
        )
        nodes = [index.tree.all_nodes[row["node_index"]] for row in info]
        layer_of = {row["node_index"]: row["layer_number"] for row in info}
        return self._result(RAPTOR_ORIGINAL_ID, nodes, layer_of, context, started)


__all__ = [
    "RAPTOR_ORIGINAL_ID",
    "ConfigRef",
    "RetrievalExecutor",
    "RetrievalResult",
    "Strategy",
]
