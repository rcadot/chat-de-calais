# tests/test_ocr.py
"""Tests de l'OCR des PDF scannés et des corrections du pipeline associées."""

import io
import os
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

import config
import loaders


@pytest.fixture
def ocr_env(monkeypatch, temp_dir):
    """Active l'OCR avec un cache isolé dans un dossier temporaire."""
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(config, "OCR_ENGINE", "openweight")
    monkeypatch.setattr(config, "OCR_CACHE_DIR", os.path.join(temp_dir, "ocr_cache"))
    return temp_dir


@pytest.fixture
def mixed_pdf(temp_dir):
    """PDF de 2 pages : page 1 avec texte, page 2 sans couche texte (scannée)."""
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    path = os.path.join(temp_dir, "mixte.pdf")
    c = canvas.Canvas(path)
    c.drawString(100, 750, "Page texte : objectif ZAN et sobriete fonciere " * 2)
    c.showPage()
    c.rect(100, 600, 200, 100, fill=1)  # page « image », sans texte
    c.showPage()
    c.save()
    return path


# ==================== DÉTECTION ====================


def test_is_scanned_page():
    """Une page est scannée en dessous du seuil de caractères utiles."""
    assert loaders.is_scanned_page("")
    assert loaders.is_scanned_page("  \n 12 \n ")
    assert not loaders.is_scanned_page("x" * config.OCR_MIN_CHARS)


# ==================== PDF ====================


def test_load_pdf_ocr_only_scanned_pages(ocr_env, mixed_pdf):
    """Seule la page sans texte part à l'OCR ; métadonnées page et ocr renseignées."""
    with patch("loaders.albert_client.ocr_image", return_value="Texte OCR page 2") as ocr:
        docs = loaders.load_document(mixed_pdf)

    assert ocr.call_count == 1
    png = ocr.call_args.args[0]
    assert png.startswith(b"\x89PNG")

    assert len(docs) == 2
    assert docs[0].metadata["ocr"] is False
    assert docs[1].metadata["ocr"] is True
    assert docs[1].metadata["page"] == 1
    assert docs[1].metadata["ocr_model"] == "openweight-ocr"
    assert docs[1].page_content == "Texte OCR page 2"


def test_load_pdf_uses_cache(ocr_env, mixed_pdf):
    """Un second chargement relit le cache sans rappeler l'OCR."""
    with patch("loaders.albert_client.ocr_image", return_value="Texte OCR"):
        loaders.load_pdf(mixed_pdf)
    with patch("loaders.albert_client.ocr_image") as ocr:
        docs = loaders.load_pdf(mixed_pdf)

    ocr.assert_not_called()
    assert docs[1].page_content == "Texte OCR"


def test_load_pdf_ocr_disabled(ocr_env, mixed_pdf, monkeypatch):
    """OCR désactivé : aucun appel, comportement PyPDF d'origine."""
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    with patch("loaders.albert_client.ocr_image") as ocr:
        docs = loaders.load_pdf(mixed_pdf)

    ocr.assert_not_called()
    assert all(d.metadata["ocr"] is False for d in docs)


def test_load_pdf_ocr_failure_drops_empty_page(ocr_env, mixed_pdf):
    """Échec de l'OCR : la page vide est écartée, la page texte conservée."""
    with patch("loaders.albert_client.ocr_image", side_effect=RuntimeError("503")):
        docs = loaders.load_pdf(mixed_pdf)

    assert len(docs) == 1
    assert docs[0].metadata["page"] == 0


def test_mistral_falls_back_to_openweight(ocr_env, mixed_pdf, monkeypatch):
    """Moteur Mistral refusé (accès restreint) : repli sur le modèle ouvert."""
    monkeypatch.setattr(config, "OCR_ENGINE", "mistral")
    with patch(
        "loaders.albert_client.ocr_document_mistral", side_effect=RuntimeError("403")
    ) as mistral, patch(
        "loaders.albert_client.ocr_image", return_value="Texte repli"
    ) as ocr:
        docs = loaders.load_pdf(mixed_pdf)

    mistral.assert_called_once()
    assert mistral.call_args.kwargs["pages"] == [1]
    ocr.assert_called_once()
    assert docs[1].page_content == "Texte repli"


@pytest.fixture
def text_page_with_images(temp_dir):
    """PDF d'une page de texte avec un grand schéma et un petit logo."""
    pytest.importorskip("reportlab")
    from PIL import Image
    from reportlab.pdfgen import canvas

    img_path = os.path.join(temp_dir, "schema.png")
    Image.new("RGB", (400, 300), "gray").save(img_path)

    path = os.path.join(temp_dir, "texte_et_image.pdf")
    c = canvas.Canvas(path)  # A4 portrait par défaut : 595 x 842 pt
    c.drawString(72, 800, "Texte natif de la page sur la sobriete fonciere " * 2)
    c.drawImage(img_path, 72, 300, width=400, height=300)  # environ 24 % de la page
    c.drawImage(img_path, 500, 780, width=40, height=30)  # logo, moins de 1 %
    c.showPage()
    c.save()
    return path


def test_find_image_regions_ignores_small_logos(ocr_env, text_page_with_images):
    """Seul le grand schéma est retenu ; le logo est écarté."""
    regions = loaders.find_image_regions(text_page_with_images, [0])
    assert len(regions[0]) == 1
    left, bottom, right, top = regions[0][0]
    assert round(left) == 72 and round(top) == 600


def test_text_page_images_are_ocrized(ocr_env, text_page_with_images):
    """Le texte de l'image est ajouté au texte natif de la page."""
    with patch(
        "loaders.albert_client.ocr_image",
        return_value="Tableau : consommation d'espaces 2011-2021 par commune",
    ) as ocr:
        docs = loaders.load_pdf(text_page_with_images)

    assert ocr.call_count == 1
    assert "Texte natif" in docs[0].page_content
    assert "[Texte extrait d'une image]" in docs[0].page_content
    assert docs[0].metadata["ocr_images"] == 1
    assert docs[0].metadata["ocr"] is False  # le texte natif reste la base


def test_text_page_images_without_text_are_ignored(ocr_env, text_page_with_images):
    """Une image sans texte (photo) n'ajoute rien à la page."""
    with patch("loaders.albert_client.ocr_image", return_value=""):
        docs = loaders.load_pdf(text_page_with_images)

    assert "[Texte extrait d'une image]" not in docs[0].page_content
    assert "ocr_images" not in docs[0].metadata


# ==================== IMAGES ====================


def test_load_image(ocr_env):
    """Une image est transcrite et produit un document page 0."""
    from PIL import Image

    path = os.path.join(ocr_env, "scan.jpg")
    Image.new("RGB", (3000, 2000), "white").save(path)

    with patch("loaders.albert_client.ocr_image", return_value="Arrêté du 12 mars") as ocr:
        docs = loaders.load_document(path)

    sent = Image.open(io.BytesIO(ocr.call_args.args[0]))
    assert max(sent.size) <= config.OCR_MAX_SIDE
    assert docs[0].page_content == "Arrêté du 12 mars"
    assert docs[0].metadata["ocr"] is True


# ==================== PIPELINE ====================


def test_format_context_includes_source_and_page():
    """Le contexte transmis au LLM porte le nom du document et la page (base 1)."""
    from rag_pipeline import format_context

    context = format_context(
        [
            Document(page_content="A", metadata={"source": "/x/note.pdf", "page": 2}),
            Document(page_content="B", metadata={"source": "/x/fiche.odt"}),
        ]
    )
    assert "[1] note.pdf, p. 3\nA" in context
    assert "[2] fiche.odt\nB" in context


def test_generate_hyde_uses_requested_mode(monkeypatch):
    """HyDE utilise le mode demandé et non le mode global."""
    import rag_pipeline

    monkeypatch.setattr(config, "USE_HYDE", True)
    spy = Mock(return_value="{query}")
    monkeypatch.setattr(config, "get_prompt_template", spy)

    rag_pipeline.generate_hyde("question", Mock(), mode="technique")

    spy.assert_called_with("hyde", mode="technique")


def test_combined_retriever_interleaves():
    """Le retriever combiné entrelace base permanente et documents temporaires."""
    from temp_documents import CombinedRetriever

    perm = Mock()
    perm.invoke.return_value = ["p1", "p2", "p3"]
    temp = Mock()
    temp.invoke.return_value = ["t1"]

    assert CombinedRetriever([perm, None, temp]).invoke("q") == ["p1", "t1", "p2", "p3"]
