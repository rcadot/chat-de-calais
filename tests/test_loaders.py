# tests/test_loaders.py
"""Tests des chargeurs de documents (hors OCR, testé dans test_ocr.py)."""

import os
from email.message import EmailMessage
from pathlib import Path

import pytest

import loaders
from indexer import split_documents

FIXTURES = Path(__file__).parent / "fixtures"


def test_txt_latin1(temp_dir):
    """Un texte en latin-1 est lu malgré l'échec de l'UTF-8."""
    path = os.path.join(temp_dir, "ancien.txt")
    Path(path).write_bytes("Arrêté préfectoral".encode("latin-1"))
    docs = loaders.load_document(path)
    assert docs[0].page_content == "Arrêté préfectoral"


def test_markdown_read_as_text(temp_dir):
    """Le Markdown est lu tel quel, titres compris."""
    path = os.path.join(temp_dir, "note.md")
    Path(path).write_text("# Titre\n\nContenu sur le ZAN.", encoding="utf-8")
    docs = loaders.load_document(path)
    assert "# Titre" in docs[0].page_content
    assert "ZAN" in docs[0].page_content


def test_docx(temp_dir):
    """Un .docx est lu."""
    docx = pytest.importorskip("docx")
    path = os.path.join(temp_dir, "note.docx")
    document = docx.Document()
    document.add_paragraph("Contenu DOCX sur le ZAN.")
    document.save(path)
    assert "ZAN" in loaders.load_document(path)[0].page_content


def test_odt_headings_kept_footer_ignored(temp_dir):
    """ODT : titres conservés dans l'ordre, pied de page (adresse) écarté."""
    from odf.opendocument import OpenDocumentText
    from odf.style import Footer, MasterPage, PageLayout
    from odf.text import H, P

    document = OpenDocumentText()
    layout = PageLayout(name="Mise")
    document.automaticstyles.addElement(layout)
    master = MasterPage(name="Standard", pagelayoutname=layout)
    footer = Footer()
    footer.addElement(P(text="100 Avenue Winston Churchill"))
    master.addElement(footer)
    document.masterstyles.addElement(master)
    document.text.addElement(H(outlinelevel=1, text="1. Contexte"))
    document.text.addElement(P(text="La loi fixe une trajectoire."))
    path = os.path.join(temp_dir, "note.odt")
    document.save(path)

    content = loaders.load_document(path)[0].page_content
    assert content == "1. Contexte\nLa loi fixe une trajectoire."


def test_html(temp_dir):
    """Une page HTML est lue."""
    pytest.importorskip("unstructured")
    path = os.path.join(temp_dir, "page.html")
    Path(path).write_text("<html><body><p>Contenu sur le ZAN.</p></body></html>", encoding="utf-8")
    assert "ZAN" in loaders.load_document(path)[0].page_content


def test_doc_word97():
    """Un .doc Word 97-2003 est lu sans LibreOffice : accents, champ, tableau."""
    content = loaders.load_document(str(FIXTURES / "exemple.doc"))[0].page_content
    assert content.startswith("Note relative à la sobriété foncière")
    assert "réduite de 50 %" in content
    assert "DATE" not in content  # code du champ supprimé, résultat conservé
    assert "Commune\tHectares\nCalais\t211" in content


def test_doc_not_ole_returns_empty(temp_dir):
    """Un faux .doc (texte renommé) est ignoré sans erreur."""
    path = os.path.join(temp_dir, "faux.doc")
    Path(path).write_text("pas un document Word", encoding="utf-8")
    assert loaders.load_document(path) == []


def test_eml_body_and_attachments(temp_dir):
    """Courriel : en-têtes et corps, pièce jointe chargée, image intégrée ignorée."""
    message = EmailMessage()
    message["Subject"] = "Intervention suite aux régulations"
    message["From"] = "agent@example.fr"
    message["Date"] = "Mon, 18 Nov 2024 10:00:00 +0100"
    message.set_content("Bonjour,\nveuillez trouver la note en pièce jointe.")
    message.add_alternative("<p>Bonjour, <b>version HTML</b></p>", subtype="html")
    message.get_payload()[1].add_related(b"\x89PNG...", maintype="image", subtype="png", cid="sig")
    message.add_attachment(
        "Note : la population de lapins a doublé.".encode("utf-8"),
        maintype="text",
        subtype="plain",
        filename="note.txt",
    )
    path = os.path.join(temp_dir, "courriel.eml")
    Path(path).write_bytes(bytes(message))

    docs = loaders.load_document(path)

    assert len(docs) == 2
    assert docs[0].page_content.startswith("Objet : Intervention suite aux régulations")
    assert "veuillez trouver la note" in docs[0].page_content  # texte brut préféré
    assert docs[1].metadata == {"source": path, "attachment": "note.txt"}
    assert "lapins" in docs[1].page_content


def test_unsupported_extension(temp_dir):
    """Une extension inconnue ne produit aucun document."""
    path = os.path.join(temp_dir, "data.xyz")
    Path(path).write_text("contenu", encoding="utf-8")
    assert loaders.load_document(path) == []


def test_split_documents_numbers_chunks():
    """Les chunks d'un fichier sont numérotés dans l'ordre de lecture."""
    from langchain_core.documents import Document

    pages = [Document(page_content="mot " * 800, metadata={"source": "a.pdf", "page": p}) for p in range(2)]
    chunks = split_documents(pages)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert len(chunks) > 2


def test_odt_ignores_comments_and_deleted_text(temp_dir):
    """ODT : commentaires de relecture et texte supprimé (suivi des modifications) exclus."""
    from odf import office, text as odf_text
    from odf.opendocument import OpenDocumentText

    document = OpenDocumentText()
    tracked = odf_text.TrackedChanges()
    from odf.namespaces import XMLNS

    region = odf_text.ChangedRegion(check_grammar=False)
    region.setAttrNS(XMLNS, "id", "c1")  # attribut xml:id obligatoire
    deletion = odf_text.Deletion()
    deletion.addElement(office.ChangeInfo())
    deletion.addElement(odf_text.P(text="Ancienne phrase supprimée."))
    region.addElement(deletion)
    tracked.addElement(region)
    document.text.addElement(tracked)
    paragraph = odf_text.P(text="Texte final de la note.")
    annotation = office.Annotation()
    annotation.addElement(odf_text.P(text="Relecteur : inverser les deux points ?"))
    paragraph.addElement(annotation)
    document.text.addElement(paragraph)
    path = os.path.join(temp_dir, "relue.odt")
    document.save(path)

    assert loaders.load_document(path)[0].page_content == "Texte final de la note."


def test_ocr_cleanup_loops_and_prompt_echo():
    """OCR : boucles de répétition réduites, consigne recopiée retirée."""
    import config

    looped = "Légende\n" + "- CCA\n" * 300 + "Total 40 059 logts"
    assert loaders.normalize_ocr_text(looped) == "Légende\n- CCA\n- CCA\nTotal 40 059 logts"
    echoed = config.OCR_PROMPT + "\n| NOM | ADRESSE |"
    assert loaders.normalize_ocr_text(echoed) == "| NOM | ADRESSE |"
