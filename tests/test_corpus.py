# tests/test_corpus.py
"""Tests des règles pour corpus à documents similaires (corpus.py)."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import config
import corpus
from indexer import split_documents
from retrieval import BM25Index, HybridRetriever

# Chemins réels du corpus (seul le nom compte, les fichiers ne sont pas lus)
DRAFT = "documents/2024/08 04 2024/Note Préfet PN -20240404.odt"
VALIDATED = "documents/2024/08 04 2024/version validée/5- Note Préfet PN -20240404.pdf"
LATER = "documents/2024/09 09 2024/Version finale/2- note pft ZAENR.pdf"


def _doc(text, path, rank_score=None, **meta):
    """Passage de test avec métadonnées de corpus."""
    metadata = {"source": path, **corpus.document_metadata(path), **meta}
    if rank_score is not None:
        metadata["rerank_score"] = rank_score
    return Document(page_content=text, metadata=metadata)


# ==================== MÉTADONNÉES ====================


def test_meeting_date_from_folder():
    """Date tirée du dossier de réunion le plus proche, 0 si absente."""
    assert corpus.meeting_date(VALIDATED) == 20240408
    assert corpus.meeting_date(LATER) == 20240909
    assert corpus.meeting_date("documents/sans_date/note.pdf") == 0
    assert corpus.format_meeting_date(20240408) == "08/04/2024"
    assert corpus.format_meeting_date(0) == ""


def test_validated_status():
    """Version validée, finale, définitive ou signée détectée sans accents ni casse."""
    assert corpus.is_validated(VALIDATED)
    assert corpus.is_validated(LATER)
    assert corpus.is_validated("x/13 05 2024/Version définitive/ODJ.pdf")
    assert corpus.is_validated("x/4 - Instruction datée et signée TREL.pdf")
    assert not corpus.is_validated(DRAFT)


def test_title_and_family():
    """Brouillon et version validée d'une même note : même famille."""
    assert corpus.document_title(VALIDATED) == "Note Préfet PN -20240404"
    assert corpus.document_family(DRAFT) == corpus.document_family(VALIDATED)
    assert corpus.document_family("a/3- PAR V2.odt") == corpus.document_family("a/PAR V2.pdf") == "par"


def test_exclusion_patterns(monkeypatch):
    """Les motifs de documents.exclure écartent les fichiers (insensible à la casse)."""
    monkeypatch.setattr(config, "DOCUMENT_EXCLUDE_PATTERNS", ["odj*"])
    assert corpus.is_excluded("x/ODJ.pdf")
    assert not corpus.is_excluded(DRAFT)


# ==================== EN-TÊTE CONTEXTUEL ====================


def test_context_header_added_and_removable(monkeypatch):
    """L'en-tête est indexé avec le passage mais retiré à l'affichage."""
    monkeypatch.setattr(config, "CONTEXT_HEADER", True)
    chunks = split_documents([Document(page_content="Texte de la note.", metadata={"source": VALIDATED})])

    chunk = chunks[0]
    assert chunk.page_content.startswith(
        "[Note Préfet PN -20240404 | réunion du 08/04/2024 | version validée]\n"
    )
    assert corpus.chunk_body(chunk) == "Texte de la note."
    assert chunk.metadata["meeting_date"] == 20240408 and chunk.metadata["validated"] is True


def test_context_header_disabled(monkeypatch):
    """Sans en-tête : texte inchangé, header_len nul."""
    monkeypatch.setattr(config, "CONTEXT_HEADER", False)
    chunk = split_documents([Document(page_content="Texte.", metadata={"source": DRAFT})])[0]
    assert chunk.page_content == "Texte." and chunk.metadata["header_len"] == 0


# ==================== DOUBLONS ET PLAFOND ====================


def test_collapse_prefers_validated_then_recent():
    """Passages quasi identiques : la version validée la plus récente représente le groupe."""
    text = "Le diagnostic des passages à niveau a été réalisé sur les voies communales en 2022 " * 3
    draft = _doc(text, DRAFT)
    validated = _doc(text + " fin", VALIDATED)
    other = _doc("Un tout autre sujet : la zone d'accélération des énergies renouvelables.", LATER)

    result = corpus.collapse_duplicates([draft, other, validated], threshold=0.8)

    assert [d.metadata["source"] for d in result] == [VALIDATED, LATER]
    assert "Note Préfet PN -20240404.odt" in result[0].metadata["aussi_dans"]
    assert "aussi_dans" not in draft.metadata  # copies, originaux intacts


def test_cap_per_document():
    """Au plus N passages par document (famille et réunion)."""
    docs = [_doc(f"p{i}", VALIDATED) for i in range(3)] + [_doc("autre", LATER)]
    kept = corpus.cap_per_document(docs, max_per_document=2)
    assert [d.page_content for d in kept] == ["p0", "p1", "autre"]
    assert len(corpus.cap_per_document(docs, max_per_document=0)) == 4


# ==================== FILTRES ====================


def test_build_where_and_matches():
    """Clause Chroma et filtre en mémoire suivent la même logique."""
    filters = {"date_min": 20240401, "date_max": 20240430, "validated_only": True}
    assert corpus.build_where(filters) == {
        "$and": [
            {"meeting_date": {"$gte": 20240401}},
            {"meeting_date": {"$lte": 20240430}},
            {"validated": True},
        ]
    }
    assert corpus.build_where({"validated_only": True}) == {"validated": True}
    assert corpus.build_where({}) is None
    assert corpus.matches_filters({"meeting_date": 20240408, "validated": True}, filters)
    assert not corpus.matches_filters({"meeting_date": 20240408, "validated": False}, filters)
    assert not corpus.matches_filters({"meeting_date": 20240909, "validated": True}, filters)


def test_hybrid_filters_bm25_and_vector():
    """Filtres : BM25 restreint, recherche vectorielle passée par where Chroma."""
    docs = [_doc("passage à niveau diagnostic", DRAFT), _doc("passage à niveau validé", VALIDATED)]
    vector = MagicMock()
    vector.vectorstore.similarity_search.return_value = []
    retriever = HybridRetriever(vector, BM25Index(docs))

    result = retriever.invoke("q", keyword_query="passage niveau", filters={"validated_only": True})

    assert [d.metadata["source"] for d in result] == [VALIDATED]
    _, kwargs = vector.vectorstore.similarity_search.call_args
    assert kwargs["filter"] == {"validated": True}
    vector.invoke.assert_not_called()
    assert retriever.meeting_dates() == [20240408]


def test_near_duplicate_handles_shifted_chunks():
    """Même texte découpé différemment (PDF/ODT) : doublon par inclusion ; fragment court : non."""
    base = " ".join(f"mot{i}" for i in range(200))
    longer = base + " " + " ".join(f"suite{i}" for i in range(120))
    a, b = corpus.shingles(base), corpus.shingles(longer)
    assert corpus.jaccard(a, b) < 0.85
    assert corpus.near_duplicate(a, b, threshold=0.85, min_size=80)
    short = corpus.shingles("100 avenue Winston Churchill CS 10007 62022 Arras Tél 03 21 22 99 99")
    page = corpus.shingles("100 avenue Winston Churchill CS 10007 62022 Arras Tél 03 21 22 99 99 " + base)
    assert not corpus.near_duplicate(short, page, threshold=0.85, min_size=80)
