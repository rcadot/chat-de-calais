# tests/test_pipeline_qualite.py
"""Tests des améliorations de qualité du pipeline : questions de suivi, seuil
de pertinence, contexte élargi, citations numérotées, rerank sans effet de bord."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import config
import rag_pipeline
from retrieval import BM25Index, HybridRetriever, merge_overlapping


def _chunk(text, index, source="/docs/note.pdf", page=0, **meta):
    """Chunk numéroté de test."""
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "chunk_index": index, **meta},
    )


@pytest.fixture
def no_hyde_no_rerank(monkeypatch):
    """Pipeline minimal : sans HyDE ni rerank."""
    monkeypatch.setattr(config, "USE_HYDE", False)
    monkeypatch.setattr(config, "USE_RERANK", False)


# ==================== QUESTIONS DE SUIVI ====================


def test_condense_without_history_returns_query():
    """Sans historique, la question n'est pas réécrite (aucun appel LLM)."""
    llm = MagicMock()
    assert rag_pipeline.condense_question("Et Calais ?", [], llm) == "Et Calais ?"
    llm.invoke.assert_not_called()


def test_condense_with_history_rewrites_question():
    """Avec historique, la question de suivi devient autonome."""
    llm = FakeListChatModel(responses=["Quelle surface a été consommée à Calais ?"])
    history = [
        {"role": "user", "content": "Quelle surface a été consommée à Arras ?"},
        {"role": "assistant", "content": "152 ha [1]"},
    ]
    result = rag_pipeline.condense_question("Et à Calais ?", history, llm)
    assert result == "Quelle surface a été consommée à Calais ?"


def test_condense_failure_falls_back_to_query():
    """Si le LLM échoue, la question d'origine est conservée."""
    llm = FakeListChatModel(responses=[])  # lève une erreur à l'appel
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    assert rag_pipeline.condense_question("Et Calais ?", history, llm) == "Et Calais ?"


# ==================== RERANK ET SEUIL ====================


def test_rerank_sorts_and_does_not_mutate(monkeypatch):
    """Le rerank trie par score et renvoie des copies (index BM25 partagé intact)."""
    monkeypatch.setattr(config, "USE_RERANK", True)
    docs = [_chunk("A", 0), _chunk("B", 1), _chunk("C", 2)]
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [{"index": 0, "score": 0.1}, {"index": 2, "score": 0.9}, {"index": 1, "score": 0.0}]
    }
    with patch("rag_pipeline.requests.post", return_value=response):
        result = rag_pipeline.rerank_documents("q", docs, top_k=3)

    assert [d.page_content for d in result] == ["C", "A", "B"]
    assert result[0].metadata["rerank_score"] == 0.9
    assert all("rerank_score" not in d.metadata for d in docs)


def test_filter_relevant_threshold(monkeypatch):
    """Passages sous le seuil écartés ; passages sans score conservés."""
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", 0.05)
    docs = [
        _chunk("A", 0, rerank_score=0.4),
        _chunk("B", 1, rerank_score=0.01),
        _chunk("C", 2),
    ]
    assert [d.page_content for d in rag_pipeline.filter_relevant(docs)] == ["A", "C"]


def test_stream_no_relevant_docs_skips_generation(monkeypatch):
    """Rien au-dessus du seuil : message explicite, pas d'appel au LLM."""
    monkeypatch.setattr(config, "USE_HYDE", False)
    monkeypatch.setattr(config, "USE_RERANK", True)
    retriever = MagicMock(spec=["invoke"])
    retriever.invoke.return_value = [_chunk("hors sujet", 0)]
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": [{"index": 0, "score": 0.0}]}
    llm = MagicMock()

    with patch("rag_pipeline.requests.post", return_value=response):
        items = list(rag_pipeline.rag_query_stream("tarte au sucre ?", retriever, llm))

    text = "".join(i["content"] for i in items if i["type"] == "chunk")
    metadata = items[-1]
    assert text == config.NO_ANSWER_MESSAGE
    assert metadata["no_relevant_docs"] is True
    assert metadata["passages"] == []
    llm.stream.assert_not_called()


# ==================== CONTEXTE ÉLARGI ET CITATIONS ====================


def test_merge_overlapping_removes_duplicate_overlap():
    """Le recouvrement entre chunks consécutifs n'apparaît qu'une fois."""
    assert merge_overlapping(["Le ZAN impose", "impose une trajectoire"]) == (
        "Le ZAN impose une trajectoire"
    )
    assert merge_overlapping(["Bloc A.", "Bloc B."]) == "Bloc A.\nBloc B."


def test_neighbors_expand_context(no_hyde_no_rerank):
    """Le contexte d'un passage inclut ses chunks voisins, dans l'ordre."""
    chunks = [_chunk("Avant.", 0), _chunk("Passage central.", 1), _chunk("Après.", 2)]
    vector = MagicMock(spec=["invoke"])
    vector.invoke.return_value = [chunks[1]]
    retriever = HybridRetriever(vector, BM25Index(chunks))

    state = rag_pipeline.retrieve_passages("question", retriever, MagicMock(), top_k=1)

    assert "Avant.\nPassage central.\nAprès." in state["context"]
    assert state["docs_final"][0].page_content == "Passage central."  # extrait affiché


def test_numbered_citations_match_passages(no_hyde_no_rerank):
    """Les numéros du contexte et ceux des passages affichés coïncident."""
    docs = [_chunk("A", 0, page=4), _chunk("B", 0, source="/docs/mail.eml", page=None, attachment="annexe.pdf")]
    retriever = MagicMock(spec=["invoke"])
    retriever.invoke.return_value = docs
    llm = FakeListChatModel(responses=["Réponse [1][2]"])

    result = rag_pipeline.rag_query("question", retriever, llm)

    assert result["answer"] == "Réponse [1][2]"
    labels = [(p["num"], p["label"]) for p in result["passages"]]
    assert labels == [(1, "note.pdf, p. 5"), (2, "mail.eml > annexe.pdf")]


def test_stream_emits_status_then_chunks(no_hyde_no_rerank):
    """Le flux annonce les étapes avant le texte de la réponse."""
    retriever = MagicMock(spec=["invoke"])
    retriever.invoke.return_value = [_chunk("A", 0)]
    llm = FakeListChatModel(responses=["ok"])

    types = [i["type"] for i in rag_pipeline.rag_query_stream("q", retriever, llm)]

    assert types[0] == "status"
    assert "chunk" in types
    assert types[-1] == "metadata"


# ==================== ÉTAPES, DOUBLONS, FILTRES ====================


def test_status_events_describe_steps(no_hyde_no_rerank):
    """Le flux annonce les étapes avec des détails chiffrés et le temps écoulé."""
    retriever = MagicMock(spec=["invoke"])
    retriever.invoke.return_value = [_chunk("A", 0), _chunk("B", 1, source="/docs/autre.pdf")]
    llm = FakeListChatModel(responses=["Réponse [1]"])

    events = [i for i in rag_pipeline.rag_query_stream("q", retriever, llm) if i["type"] == "status"]

    steps = [e["step"] for e in events]
    assert steps[0] == "recherche" and steps[-1] == "redaction" and "tri" in steps
    details = " ".join(e.get("detail", "") for e in events)
    assert "2 passages candidats dans 2 document(s)" in details
    assert "passage(s) retenu(s)" in details
    assert all(e["elapsed"] >= 0 and e["content"] for e in events)


def test_reformulation_step_announced():
    """Une question de suivi reformulée est annoncée avec la question comprise."""
    retriever = MagicMock(spec=["invoke"])
    retriever.invoke.return_value = []
    llm = FakeListChatModel(responses=["Quel est le calendrier des ZAENR ?"])
    history = [{"role": "user", "content": "Les ZAENR ?"}, {"role": "assistant", "content": "..."}]

    events = list(rag_pipeline.iter_retrieval_steps("Et le calendrier ?", retriever, llm, history=history))

    assert events[0]["step"] == "reformulation"
    assert "« Quel est le calendrier des ZAENR ? »" in events[1]["detail"]


def test_pipeline_collapses_duplicate_versions(no_hyde_no_rerank, monkeypatch):
    """Deux versions identiques d'une note n'occupent qu'une place, la validée est citée."""
    monkeypatch.setattr(config, "DEDUP_ENABLED", True)
    text = "Le calendrier prévoit les conférences territoriales de mai à septembre 2024 " * 3
    draft = _chunk(text, 0, source="d/08 04 2024/Note.odt", meeting_date=20240408, validated=False)
    final = _chunk(text, 0, source="d/08 04 2024/version validée/Note.pdf", meeting_date=20240408, validated=True)
    retriever = MagicMock(spec=["invoke"])
    retriever.invoke.return_value = [draft, final]

    state = rag_pipeline.retrieve_passages("calendrier", retriever, MagicMock())

    assert len(state["docs_final"]) == 1
    assert state["docs_final"][0].metadata["validated"] is True
    passage = rag_pipeline.build_passages(state["docs_final"])[0]
    assert passage["aussi_dans"].startswith("Note.odt")
    assert "réunion du 08/04/2024 | version validée" in state["context"]


def test_filters_reach_hybrid_retriever(no_hyde_no_rerank):
    """Les filtres de l'interface sont transmis au retriever hybride."""
    retriever = MagicMock()
    retriever.supports_keyword_query = True
    retriever.invoke.return_value = []
    filters = {"validated_only": True}

    list(rag_pipeline.rag_query_stream("q", retriever, MagicMock(), filters=filters))

    assert retriever.invoke.call_args.kwargs["filters"] == filters


def test_ocr_html_tables_rendered_as_markdown():
    """Les tableaux HTML de l'OCR deviennent des tableaux Markdown (affichage, BM25, contexte)."""
    from loaders import normalize_ocr_text

    html = "Titre :\n<table><tr><th>EPCI</th><th>Volet<br>1</th></tr><tr><td>CA Lens</td><td>x</td></tr></table>"
    assert normalize_ocr_text(html) == "Titre :\n\n| EPCI | Volet 1 |\n| --- | --- |\n| CA Lens | x |"
    assert normalize_ocr_text("sans balise") == "sans balise"
