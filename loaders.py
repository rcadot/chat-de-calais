# loaders.py
"""Loaders pour différents formats de documents."""
import email
import email.policy
import io
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredHTMLLoader
)
import albert_client
import config

log = logging.getLogger(__name__)


# ============================================================================
# OCR
# ============================================================================


def is_scanned_page(text: str) -> bool:
    """Indique si une page PDF n'a pas de couche texte exploitable."""
    return len(re.sub(r"\s+", "", text or "")) < config.OCR_MIN_CHARS


def _html_table_to_markdown(html: str) -> str:
    """Convertit un tableau HTML (sortie du modèle d'OCR) en tableau Markdown."""
    from bs4 import BeautifulSoup

    rows = []
    for tr in BeautifulSoup(html, "html.parser").find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            for br in cell.find_all("br"):
                br.replace_with(" ")
            text = re.sub(r"\s+", " ", cell.get_text()).strip().replace("|", "\\|")
            cells.append(text)
            cells.extend([""] * (int(cell.get("colspan", 1) or 1) - 1))
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def normalize_ocr_text(text: str) -> str:
    """
    Nettoie le texte produit par l'OCR : tableaux HTML convertis en Markdown,
    sauts de ligne et exposants HTML remplacés.

    Les balises encombreraient la recherche par mots-clés (td, tr...), le
    contexte du modèle et l'affichage des extraits.

    Args:
        text: Texte brut de l'OCR

    Returns:
        str: Texte en Markdown
    """
    if not text:
        return ""
    # Consigne recopiée par le modèle d'OCR
    text = text.replace(config.OCR_PROMPT, "")
    if "<" in text:
        text = re.sub(
            r"<table\b.*?</table>",
            lambda m: "\n\n" + _html_table_to_markdown(m.group(0)) + "\n\n",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</?sup>", "", text, flags=re.IGNORECASE)
    # Boucles de l'OCR (même ligne répétée des dizaines de fois) : 2 occurrences gardées
    text = re.sub(r"(^[^\n]+\n)(?:\1){2,}", r"\1\1", text + "\n", flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _to_png(image, max_side: int) -> bytes:
    """Réduit une image PIL au côté maximal demandé et l'encode en PNG."""
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_pdf_page(
    file_path: str,
    page_index: int,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_side: Optional[int] = None,
) -> bytes:
    """
    Rend une page PDF (ou une zone de la page) en PNG.

    Args:
        file_path: Chemin du PDF
        page_index: Indice de page (base 0)
        bounds: Zone à extraire (gauche, bas, droite, haut) en points PDF,
            None pour la page entière
        max_side: Côté maximal de l'image (défaut : OCR_MAX_SIDE)

    Returns:
        bytes: Image PNG
    """
    import pypdfium2 as pdfium

    max_side = max_side or config.OCR_MAX_SIDE
    pdf = pdfium.PdfDocument(file_path)
    try:
        page = pdf[page_index]
        width, height = page.get_size()  # en points (1/72 pouce)
        if bounds is None:
            scale = min(config.OCR_DPI / 72, max_side / max(width, height))
        else:
            scale = config.OCR_DPI / 72  # pleine résolution avant découpe
        image = page.render(scale=scale).to_pil()
        if bounds is not None:
            left, bottom, right, top = bounds
            image = image.crop(
                (
                    int(left * scale),
                    int((height - top) * scale),
                    int(right * scale),
                    int((height - bottom) * scale),
                )
            )
        return _to_png(image, max_side)
    finally:
        pdf.close()


def _ocr_cache_file(file_path: str) -> Path:
    """Chemin du cache OCR d'un fichier (clé : hash du contenu et moteur)."""
    from indexer import get_file_hash  # import local : indexer importe loaders

    key = f"{get_file_hash(file_path)}_{config.OCR_ENGINE}"
    return Path(config.OCR_CACHE_DIR) / f"{key}.json"


def _read_ocr_cache(file_path: str) -> Dict[str, str]:
    """
    Lit le cache OCR d'un fichier (dictionnaire vide si absent).

    Clés : "<page>" pour une page entière, "<page>_img<k>" pour une image
    insérée dans une page de texte.
    """
    cache_file = _ocr_cache_file(file_path)
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, encoding="utf-8") as f:
            return {str(k): v for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return {}


def _write_ocr_cache(file_path: str, cache: Dict[str, str]) -> None:
    """Écrit le cache OCR d'un fichier."""
    cache_file = _ocr_cache_file(file_path)
    os.makedirs(cache_file.parent, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def ocr_pdf_pages(file_path: str, page_indices: List[int]) -> Dict[int, str]:
    """
    Transcrit des pages PDF par OCR, avec cache disque.

    Le moteur Mistral traite le PDF en une requête ; en cas d'échec (accès
    refusé, taille), on se replie sur le modèle ouvert page par page.

    Args:
        file_path: Chemin du PDF
        page_indices: Indices des pages à transcrire (base 0)

    Returns:
        Dict[int, str]: Texte par indice de page
    """
    cache = _read_ocr_cache(file_path)
    todo = [i for i in page_indices if str(i) not in cache]

    if todo and config.OCR_ENGINE == "mistral":
        try:
            pdf_bytes = Path(file_path).read_bytes()
            result = albert_client.ocr_document_mistral(pdf_bytes, pages=todo)
            cache.update({str(i): text for i, text in result.items()})
            todo = [i for i in todo if str(i) not in cache]
        except Exception as e:
            log.warning("OCR Mistral indisponible (%s), repli sur openweight", e)

    for i in todo:
        try:
            cache[str(i)] = albert_client.ocr_image(render_pdf_page(file_path, i))
            log.info("   🔎 OCR %s p.%d", Path(file_path).name, i + 1)
        except Exception as e:
            log.warning("OCR %s p.%d: %s", Path(file_path).name, i + 1, e)

    _write_ocr_cache(file_path, cache)
    return {i: normalize_ocr_text(cache[str(i)]) for i in page_indices if str(i) in cache}


def find_image_regions(
    file_path: str, page_indices: List[int]
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Repère les images significatives des pages indiquées.

    Une image est retenue si elle couvre entre OCR_IMAGE_MIN_RATIO (on écarte
    logos et pictogrammes) et OCR_IMAGE_MAX_RATIO de la page (au-delà, la page
    est un scan déjà doté d'une couche texte : le réOCRiser ferait doublon).

    Args:
        file_path: Chemin du PDF
        page_indices: Indices des pages à examiner (base 0)

    Returns:
        Dict[int, list]: Pour chaque page ayant des images retenues, leurs
        bornes (gauche, bas, droite, haut) en points PDF
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    regions = {}
    pdf = pdfium.PdfDocument(file_path)
    try:
        for i in page_indices:
            page = pdf[i]
            width, height = page.get_size()
            kept = []
            for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
                left, bottom, right, top = obj.get_bounds()
                left, right = max(0.0, left), min(width, right)
                bottom, top = max(0.0, bottom), min(height, top)
                ratio = max(0.0, right - left) * max(0.0, top - bottom) / (width * height)
                if config.OCR_IMAGE_MIN_RATIO <= ratio < config.OCR_IMAGE_MAX_RATIO:
                    kept.append((left, bottom, right, top))
            if kept:
                regions[i] = kept
    finally:
        pdf.close()
    return regions


def ocr_image_regions(file_path: str, page_indices: List[int]) -> Dict[int, List[str]]:
    """
    Transcrit le texte des images insérées dans des pages qui ont déjà du texte
    (schémas, tableaux scannés, annexes collées), avec cache disque.

    Args:
        file_path: Chemin du PDF
        page_indices: Indices des pages à examiner (base 0)

    Returns:
        Dict[int, List[str]]: Textes non vides extraits, par page
    """
    regions = find_image_regions(file_path, page_indices)
    if not regions:
        return {}

    cache = _read_ocr_cache(file_path)
    budget = config.OCR_MAX_IMAGES
    result = {}
    for page_index, bounds_list in regions.items():
        for k, bounds in enumerate(bounds_list):
            key = f"{page_index}_img{k}"
            if key not in cache:
                if budget <= 0:
                    break
                budget -= 1
                try:
                    png = render_pdf_page(file_path, page_index, bounds)
                    cache[key] = albert_client.ocr_image(png)
                except Exception as e:
                    log.warning("OCR image %s p.%d: %s", Path(file_path).name, page_index + 1, e)
                    continue
            useful = len(re.sub(r"\s+", "", cache[key]))
            if useful >= config.OCR_IMAGE_MIN_CHARS:
                result.setdefault(page_index, []).append(normalize_ocr_text(cache[key]))

    _write_ocr_cache(file_path, cache)
    if result:
        n = sum(len(v) for v in result.values())
        log.info("   🖼️ %s: texte extrait de %d image(s)", Path(file_path).name, n)
    return result


def ocr_model_name() -> str:
    """Nom du modèle OCR configuré (pour les métadonnées)."""
    return config.OCR_MODELS.get(config.OCR_ENGINE, config.OCR_ENGINE)


def load_pdf(file_path: str) -> List[Document]:
    """
    Charge un PDF page par page, avec OCR des pages sans couche texte.

    Args:
        file_path: Chemin du PDF

    Returns:
        List[Document]: Un document par page (métadonnées source, page, ocr)
    """
    docs = PyPDFLoader(file_path).load()
    for doc in docs:
        doc.metadata["ocr"] = False

    if not config.OCR_ENABLED:
        return docs

    scanned = [d for d in docs if is_scanned_page(d.page_content)]

    # Pages de texte : OCR des images insérées (PDF mixtes)
    if config.OCR_IMAGES_IN_TEXT_PAGES:
        text_pages = [d for d in docs if not is_scanned_page(d.page_content)]
        image_texts = ocr_image_regions(
            file_path, [d.metadata.get("page", 0) for d in text_pages]
        )
        for doc in text_pages:
            texts = image_texts.get(doc.metadata.get("page", 0))
            if texts:
                doc.page_content += "".join(
                    f"\n\n[Texte extrait d'une image]\n{t}" for t in texts
                )
                doc.metadata["ocr_images"] = len(texts)
                doc.metadata["ocr_model"] = config.OCR_MODELS["openweight"]

    if not scanned:
        return docs

    if len(scanned) > config.OCR_MAX_PAGES:
        log.warning(
            "%s: %d pages scannées, OCR limité à %d",
            Path(file_path).name, len(scanned), config.OCR_MAX_PAGES,
        )
        scanned = scanned[: config.OCR_MAX_PAGES]

    log.info("   🔎 %s: OCR de %d page(s)", Path(file_path).name, len(scanned))

    texts = ocr_pdf_pages(file_path, [d.metadata.get("page", 0) for d in scanned])
    for doc in scanned:
        text = texts.get(doc.metadata.get("page", 0), "")
        if text.strip():
            doc.page_content = text
            doc.metadata["ocr"] = True
            doc.metadata["ocr_model"] = ocr_model_name()

    return [d for d in docs if d.page_content.strip()]


def load_image(file_path: str) -> List[Document]:
    """
    Charge une image (page scannée, photo de document) par OCR.

    Args:
        file_path: Chemin de l'image

    Returns:
        List[Document]: Un document, ou liste vide si OCR désactivé ou vide
    """
    if not config.OCR_ENABLED:
        return []

    cache = _read_ocr_cache(file_path)
    if "0" not in cache:
        from PIL import Image

        with Image.open(file_path) as image:
            image.seek(0)  # première page des TIFF multipages
            png = _to_png(image.copy(), config.OCR_MAX_SIDE)

        text = ""
        if config.OCR_ENGINE == "mistral":
            try:
                text = albert_client.ocr_document_mistral(png, mime="image/png").get(0, "")
            except Exception as e:
                log.warning("OCR Mistral indisponible (%s), repli sur openweight", e)
        if not text:
            text = albert_client.ocr_image(png)
        cache["0"] = text
        _write_ocr_cache(file_path, cache)

    if not cache["0"].strip():
        return []
    return [
        Document(
            page_content=normalize_ocr_text(cache["0"]),
            metadata={
                "source": file_path,
                "page": 0,
                "ocr": True,
                "ocr_model": ocr_model_name(),
            },
        )
    ]


def load_odt(file_path: str) -> List[Document]:
    """
    Charge un fichier ODT : paragraphes et titres, dans l'ordre de lecture.

    Args:
        file_path: Chemin du fichier

    Returns:
        List[Document]: Un document, ou liste vide en cas d'erreur
    """
    try:
        from odf import teletype
        from odf.namespaces import TEXTNS
        from odf.opendocument import load

        blocks = []

        def walk(node) -> None:
            """Parcourt l'arbre et collecte paragraphes et titres."""
            for child in node.childNodes:
                if child.nodeType != child.ELEMENT_NODE:
                    continue
                if child.qname in ((TEXTNS, "p"), (TEXTNS, "h")):
                    blocks.append(teletype.extractText(child))
                else:
                    walk(child)

        document = load(file_path)
        # Commentaires de relecture et texte supprimé (suivi des modifications) :
        # retirés pour ne garder que le texte final du document
        from odf import office, text as odf_text

        for cls in (
            getattr(office, "Annotation", None),
            getattr(office, "AnnotationEnd", None),
            getattr(odf_text, "TrackedChanges", None),
        ):
            if cls is None:
                continue
            for element in list(document.getElementsByType(cls)):
                element.parentNode.removeChild(element)

        walk(document.text)
        content = "\n".join(blocks)
        return [Document(page_content=content, metadata={"source": file_path})]
    except Exception as e:
        log.warning("Erreur ODT %s: %s", Path(file_path).name, e)
        return []


def load_doc(file_path: str) -> List[Document]:
    """
    Charge un fichier Word 97-2003 (.doc), sans LibreOffice ni Word.

    Args:
        file_path: Chemin du fichier

    Returns:
        List[Document]: Un document, ou liste vide en cas d'erreur
    """
    from doc_reader import extract_doc_text

    try:
        content = extract_doc_text(file_path)
    except Exception as e:
        log.warning("Erreur .doc %s: %s", Path(file_path).name, e)
        return []
    return [Document(page_content=content, metadata={"source": file_path})] if content else []


def _html_to_text(html: str) -> str:
    """Texte lisible d'un contenu HTML (corps de courriel)."""
    from bs4 import BeautifulSoup

    text = BeautifulSoup(html, "html.parser").get_text("\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def load_eml(file_path: str) -> List[Document]:
    """
    Charge un courriel (.eml) : en-têtes, corps, et pièces jointes.

    Les pièces jointes de format pris en charge sont chargées comme des
    documents à part entière (OCR compris), rattachées au courriel par
    `source` (chemin du .eml) et `attachment` (nom de la pièce jointe). Les
    images intégrées au corps (signatures, logos) sont ignorées.

    Args:
        file_path: Chemin du courriel

    Returns:
        List[Document]: Corps du courriel puis pièces jointes
    """
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)

    headers = [
        f"{label} : {msg[key]}"
        for key, label in (("subject", "Objet"), ("from", "De"), ("to", "À"), ("date", "Date"))
        if msg[key]
    ]
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = ""
    if body_part is not None:
        body = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            body = _html_to_text(body)

    docs = [
        Document(
            page_content="\n".join(headers) + "\n\n" + body.strip(),
            metadata={"source": file_path},
        )
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for part in msg.iter_attachments():
            name = part.get_filename()
            if part.get_content_disposition() != "attachment" or not name:
                continue
            ext = Path(name).suffix.lower()
            if ext not in config.SUPPORTED_EXTENSIONS or ext == ".eml":
                continue
            tmp_path = os.path.join(tmp_dir, Path(name).name)
            with open(tmp_path, "wb") as f:
                f.write(part.get_payload(decode=True) or b"")
            for doc in load_document(tmp_path):
                doc.metadata["source"] = file_path
                doc.metadata["attachment"] = name
                docs.append(doc)
            log.info("   📎 %s > %s", Path(file_path).name, name)

    return docs


def load_document(file_path: str) -> List[Document]:
    """Charge un document selon son extension."""
    ext = Path(file_path).suffix.lower()

    try:
        if ext == '.pdf':
            return load_pdf(file_path)

        elif ext in config.IMAGE_EXTENSIONS:
            return load_image(file_path)

        elif ext == '.docx':
            return Docx2txtLoader(file_path).load()

        elif ext in ['.txt', '.text', '.md']:
            # Markdown lu comme du texte : titres et listes restent lisibles
            for encoding in config.TEXT_ENCODINGS:
                try:
                    return TextLoader(file_path, encoding=encoding).load()
                except (RuntimeError, UnicodeDecodeError):
                    continue
            return []

        elif ext in ['.html', '.htm']:
            return UnstructuredHTMLLoader(file_path).load()

        elif ext == '.odt':
            return load_odt(file_path)

        elif ext == '.doc':
            return load_doc(file_path)

        elif ext == '.eml':
            return load_eml(file_path)

        return []

    except Exception as e:
        log.warning("Erreur %s: %s", Path(file_path).name, e)
        return []
