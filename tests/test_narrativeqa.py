"""NarrativeQA preprocessing and ROUGE-L (offline, no downloads)."""

from __future__ import annotations

from adaptive_raptor.data import narrativeqa as nqa
from adaptive_raptor.data.records import GoldAnswer
from adaptive_raptor.evaluation.metrics import answer_rouge_l, rouge_l_score


def test_markers_match_tokenized_forms():
    text = "Header junk.\nWHO'S YOUR DADDY\nStory body here. The end of new eBooks.\nLicense."
    story, found = nqa.cut_story(text, "WHO S YOUR", "new eBooks .")
    assert found == {"start_found": True, "end_found": True}
    assert story.startswith("WHO'S YOUR") and story.endswith("new eBooks")


def test_missing_marker_keeps_that_side():
    story, found = nqa.cut_story("alpha beta gamma", "nothing here", "gamma")
    assert story == "alpha beta gamma"
    assert found == {"start_found": False, "end_found": True}


def test_end_is_last_occurrence_after_start():
    story, _ = nqa.cut_story("x START a THE END b THE END c", "START", "THE END")
    assert story == "START a THE END b THE END"


def test_html_is_stripped_for_movies():
    raw = {
        "id": "m1",
        "kind": "movie",
        "title": "T",
        "word_count": 5,
        "start": "FADE IN",
        "end": "THE END",
        "text": "<html><script>var x=1;</script><b>FADE IN</b>:&amp; a<br>b\n\n\n\nTHE END</html>",
    }
    doc, meta = nqa.parse_document(raw)
    assert doc.text == "FADE IN :& a b\n\nTHE END"
    assert meta["start_found"] and meta["end_found"]


def test_parse_split_ids_and_answers():
    docs = [
        {
            "id": "d",
            "kind": "gutenberg",
            "title": "",
            "word_count": 3,
            "start": "a",
            "end": "c",
            "text": "a b c",
        }
    ]
    qs = [{"doc_id": "d", "index_in_doc": 0, "question": "q?", "answers": ["x", " "]}]
    _, questions, _, stats = nqa.parse_split(docs, qs)
    assert questions[0].question_id == "d:0"
    assert [a.text for a in questions[0].answers] == ["x"]
    assert stats["questions_without_answers"] == 0


def test_rouge_l():
    assert rouge_l_score("the cat sat", "the cat sat") == 1.0
    assert rouge_l_score("", "x") == 0.0
    # LCS("a b c d", "a c d e") = 3 -> P = R = 3/4
    assert abs(rouge_l_score("a b c d", "a c d e") - 0.75) < 1e-9
    golds = (GoldAnswer("black cloak", "abstractive"), GoldAnswer("the actor", "abstractive"))
    # 6 prediction tokens, LCS 2 with either reference: P = 1/3, R = 1 -> F1 = 0.5
    assert abs(answer_rouge_l("The actor in the black cloak", golds) - 0.5) < 1e-9
