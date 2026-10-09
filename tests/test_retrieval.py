# tests/test_retrieval.py
"""Tests de la recherche hybride (BM25 et fusion RRF)."""

from unittest.mock import MagicMock, Mock

from langchain_core.documents import Document

import config
from retrieval import (
    BM25Index,
    HybridRetriever,
    make_retriever,
    reciprocal_rank_fusion,
    tokenize_fr,
)


def _doc(text, source="a.pdf", page=0):
    """Construit un chunk de test."""
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_tokenize_fr_accents_stopwords_numbers():
    """Accents retirés, mots vides supprimés, nombres conservés."""
    tokens = tokenize_fr("L'article L.101-2 du Code de l'urbanisme, Cerfa 13703")
    assert "urbanisme" in tokens
    assert "101" in tokens and "13703" in tokens
    assert "de" not in tokens and "du" not in tokens


def test_bm25_finds_exact_reference():
    """BM25 retrouve un numéro de Cerfa absent du reste du corpus."""
    docs = [
        _doc("Déclaration préalable : formulaire Cerfa 13703", page=0),
        _doc("Le permis de construire concerne les extensions", page=1),
        _doc("La sobriété foncière vise le zéro artificialisation nette", page=2),
    ]
    result = BM25Index(docs).search("Quel Cerfa 13703 ?", k=5)
    assert result[0].metadata["page"] == 0
    assert len(result) == 1  # les chunks sans terme commun sont exclus


def test_bm25_empty_corpus():
    """Un corpus vide ne lève pas d'erreur."""
    assert BM25Index([]).search("test", k=5) == []


def test_rrf_favours_documents_found_by_both():
    """Un document classé par les deux listes passe devant."""
    a, b, c = _doc("A", page=0), _doc("B", page=1), _doc("C", page=2)
    fused = reciprocal_rank_fusion([[a, b], [c, b]], k=60)
    assert fused[0] is b
    assert len(fused) == 3


def test_hybrid_uses_original_question_for_keywords():
    """La question d'origine sert à BM25, la requête HyDE au vectoriel."""
    target = _doc("Cerfa 13703 déclaration préalable", page=4)
    vector = Mock()
    vector.invoke.return_value = [_doc("autre", page=9)]
    retriever = HybridRetriever(vector, BM25Index([target, _doc("divers", page=8)]))

    result = retriever.invoke("long document HyDE générique", keyword_query="Cerfa 13703")

    vector.invoke.assert_called_once_with("long document HyDE générique")
    assert target in result


def test_make_retriever_respects_flag(monkeypatch):
    """Sans recherche hybride, le retriever vectoriel est renvoyé tel quel."""
    store = MagicMock()
    monkeypatch.setattr(config, "USE_HYBRID_SEARCH", False)
    assert make_retriever(store) is store.as_retriever.return_value

    monkeypatch.setattr(config, "USE_HYBRID_SEARCH", True)
    store.get.return_value = {"documents": ["texte"], "metadatas": [{"source": "x"}]}
    assert isinstance(make_retriever(store), HybridRetriever)
