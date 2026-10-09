# tests/test_evaluate.py
"""Tests des indicateurs du script d'évaluation."""

from langchain_core.documents import Document

from evaluate import first_match_rank, load_questions, summarize


def test_first_match_rank():
    """Rang de la première source attendue, pièces jointes comprises."""
    docs = [
        Document(page_content="", metadata={"source": "/d/autre.pdf"}),
        Document(page_content="", metadata={"source": "/d/mail.eml", "attachment": "Note ZAN.pdf"}),
    ]
    assert first_match_rank(docs, ["note zan"]) == 2
    assert first_match_rank(docs, ["absent"]) is None


def test_summarize():
    """Succès, rappel, MRR et rejet des questions hors sujet."""
    common = {"n_passages": 3, "documents_distincts": 2, "doublons": 0, "duree_s": 1.0}
    rows = [
        {"sources_attendues": "a", "succes": True, "rang_recherche": 3, "rang_final": 1, **common},
        {"sources_attendues": "b", "succes": False, "rang_recherche": None, "rang_final": None, **common, "doublons": 2},
        {"sources_attendues": "(aucune)", "succes": True, "rang_recherche": None, "rang_final": None, **common},
    ]
    result = summarize(rows)
    assert result["succes_final"] == 0.5
    assert result["rappel_recherche"] == 0.5
    assert result["mrr_final"] == 0.5
    assert result["hors_sujet_rejetes"] == 1.0
    assert result["documents_distincts_moyens"] == 2
    assert round(result["doublons_moyens"], 2) == 0.67


def test_count_duplicates():
    """Deux passages identiques parmi trois : un doublon compté."""
    from evaluate import count_duplicates

    text = "le calendrier des conférences territoriales court de mai à septembre 2024"
    docs = [Document(page_content=t, metadata={}) for t in (text, "un tout autre sujet sans rapport", text)]
    assert count_duplicates(docs) == 1


def test_load_questions_template():
    """Le jeu de questions fourni est valide."""
    questions = load_questions("evaluation/questions.yaml")
    assert questions and all("question" in q for q in questions)
