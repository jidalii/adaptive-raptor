"""Answer + evidence-sufficiency prompt (cascade first stage); offline only."""

from adaptive_raptor.generation import (
    TEMPLATES,
    get_template,
    render,
    split_sufficiency,
    wants_sufficiency,
)


def test_split_sufficiency_variants():
    assert split_sufficiency("BERT and ELMo\nEvidence: SUFFICIENT") == (
        "BERT and ELMo",
        "sufficient",
    )
    assert split_sufficiency("Unanswerable\n\nevidence: insufficient.") == (
        "Unanswerable",
        "insufficient",
    )
    assert split_sufficiency("Answer: Yes\n**Evidence:** PARTIAL") == ("Yes", "partial")
    # no flag: text unchanged, sufficiency None (counted as invalid in the analysis)
    assert split_sufficiency("No") == ("No", None)
    # the word "evidence" inside an answer is not a flag
    text = "They use evidence: sufficient statistics of the model"
    assert split_sufficiency(text) == (text, None)


def test_sufficiency_template_shares_default_prefix():
    suff, default = get_template("qasper_sufficiency_v1"), get_template("qasper_default")
    assert wants_sufficiency(suff) and not wants_sufficiency(default)
    small = render(suff, "Q?", ["chunk one", "chunk two"])
    large = render(default, "Q?", ["chunk one", "chunk two", "chunk three"])
    prefix = render(default, "Q?", ["chunk one", "chunk two"]).split("\n\nQuestion:")[0]
    assert small.startswith(prefix) and large.startswith(prefix + "\n\nchunk three")
    assert set(TEMPLATES) >= {"qasper_default", "qasper_sufficiency_v1"}


def test_summary_prompt_override_keys():
    import pytest

    from adaptive_raptor.config.schema import RaptorSection
    from adaptive_raptor.indexing.builder import raptor_index_params
    from adaptive_raptor.models.summarization import UPSTREAM_TEMPLATE, summary_prompt

    default = raptor_index_params(RaptorSection())
    assert "summarization_prompt" not in default  # cached indexes keep their keys
    assert summary_prompt("upstream")[1] == UPSTREAM_TEMPLATE
    dense = raptor_index_params(RaptorSection(summarization_prompt="evidence_dense_v1"))
    assert dense["summarization_prompt"]["name"] == "evidence_dense_v1"
    assert "{context}" in summary_prompt("evidence_dense_v1")[1]
    with pytest.raises(KeyError):
        summary_prompt("nope")
