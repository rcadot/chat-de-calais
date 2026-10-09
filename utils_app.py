"""Utilitaires d'affichage de l'application de chat."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

import config


# Fonction pour formater les sources
def format_sources(sources, scores):
    """
    Formate les sources avec scores en HTML.
    Déduplique les sources et garde le meilleur score.
    """
    if not sources:
        return ""

    # Dédupliquer : garder le meilleur score pour chaque source unique
    sources_dict = {}
    for source, score in zip(sources, scores):
        source_name = Path(source).name

        if source_name not in sources_dict:
            sources_dict[source_name] = score
        else:
            # Garder le meilleur score
            if score and sources_dict[source_name]:
                sources_dict[source_name] = max(sources_dict[source_name], score)
            elif score:
                sources_dict[source_name] = score

    # Trier par score décroissant
    sorted_sources = sorted(
        sources_dict.items(), key=lambda x: x[1] if x[1] else 0, reverse=True
    )

    sources_html = '<div class="sources-section">'
    sources_html += "<strong>📚 Sources consultées :</strong><br><br>"

    for i, (source_name, score) in enumerate(sorted_sources, 1):
        if score:
            percentage = score * 100
            if score >= config.SCORE_GOOD:
                score_class = "source-score"
                emoji = "🟢"
            elif score >= config.SCORE_MEDIUM:
                score_class = "source-score source-score-medium"
                emoji = "🟡"
            else:
                score_class = "source-score source-score-low"
                emoji = "🔴"
            score_display = (
                f'<span class="{score_class}">{emoji} {percentage:.0f}%</span>'
            )
        else:
            score_display = '<span class="source-score">⚪ N/A</span>'

        sources_html += (
            f'<div class="source-item">{score_display} <code>{source_name}</code></div>'
        )

    sources_html += "</div>"
    return sources_html


# ============================================================================
# SOURCES DÉTAILLÉES (passages numérotés, aperçu, téléchargement)
# ============================================================================

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".eml": "message/rfc822",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def score_badge(score: Optional[float]) -> Tuple[str, str]:
    """
    Libellé et couleur de pastille d'un score de rerank.

    Args:
        score: Score entre 0 et 1, ou None

    Returns:
        (libellé, couleur Streamlit)
    """
    if score is None:
        return "pertinence N/A", "gray"
    if score >= config.SCORE_GOOD:
        return f"pertinence {score:.2f}", "green"
    if score >= config.SCORE_MEDIUM:
        return f"pertinence {score:.2f}", "orange"
    return f"pertinence {score:.2f}", "red"


@st.cache_data(show_spinner=False, max_entries=200)
def page_preview(file_path: str, page_index: int, mtime: float) -> bytes:
    """
    Aperçu PNG d'une page PDF (mis en cache ; mtime invalide le cache).

    Args:
        file_path: Chemin du PDF
        page_index: Indice de page (base 0)
        mtime: Date de modification du fichier (clé de cache)

    Returns:
        bytes: Image PNG
    """
    from loaders import render_pdf_page

    return render_pdf_page(file_path, page_index, max_side=1000)


def _render_passage(passage: Dict[str, Any], key: str) -> None:
    """Affiche un passage : pastilles, extrait, aperçu de la page, téléchargement."""
    label, color = score_badge(passage.get("score"))
    badges = [f":{color}-badge[{label}]"]
    if passage.get("ocr"):
        badges.append(":orange-badge[texte issu de l'OCR, à vérifier]")
    if passage.get("ocr_images"):
        badges.append(":orange-badge[contient du texte lu dans une image]")
    if passage.get("details"):
        validated = "version validée" in passage["details"]
        badges.insert(0, f":{'blue' if validated else 'gray'}-badge[{passage['details']}]")
    st.markdown(f"**[{passage['num']}] {passage['label']}**  \n" + " ".join(badges))
    if passage.get("aussi_dans"):
        st.caption(f"Passage identique aussi présent dans : {passage['aussi_dans']}")

    path = Path(passage.get("source", ""))
    exists = path.is_file()
    ext = path.suffix.lower()
    has_preview = exists and not passage.get("attachment") and (
        (ext == ".pdf" and passage.get("page") is not None) or ext in config.IMAGE_EXTENSIONS
    )

    text_col, preview_col = st.columns([3, 2]) if has_preview else (st.container(), None)
    with text_col:
        st.caption("Extrait utilisé pour la réponse")
        with st.container(border=True, height=320):
            st.markdown(passage.get("excerpt", ""))
        if exists:
            st.download_button(
                f"Ouvrir {path.name}",
                data=lambda p=path: p.read_bytes(),  # lecture à la demande
                file_name=path.name,
                mime=MIME_TYPES.get(ext, "application/octet-stream"),
                on_click="ignore",
                key=f"{key}_dl",
                icon=":material/download:",
            )
            if ext == ".pdf" and passage.get("page") is not None:
                st.caption(f"Passage en page {int(passage['page']) + 1} du document.")
        else:
            st.caption("Document temporaire ou déplacé : fichier non disponible.")

    if preview_col is not None:
        with preview_col:
            try:
                if ext == ".pdf":
                    image = page_preview(str(path), int(passage["page"]), path.stat().st_mtime)
                    st.image(image, caption=f"Page {int(passage['page']) + 1}")
                else:
                    st.image(str(path))
            except Exception as e:
                st.caption(f"Aperçu indisponible : {e}")


def _tab_label(passage: Dict[str, Any]) -> str:
    """Libellé court d'onglet : numéro, nom abrégé, page."""
    name = Path(passage.get("source", "")).stem
    name = name if len(name) <= 24 else name[:23] + "…"
    page = passage.get("page")
    return f"[{passage['num']}] {name}" + (f" p.{int(page) + 1}" if page is not None else "")


def render_passages(
    passages: List[Dict[str, Any]], key_prefix: str, no_relevant_docs: bool = False
) -> None:
    """
    Affiche les sources d'une réponse : avertissement de pertinence, puis un
    onglet par passage cité, numéroté comme dans la réponse ([1], [2]...).

    Args:
        passages: Passages décrits par rag_pipeline.build_passages
        key_prefix: Préfixe unique des clés de widgets (un par message)
        no_relevant_docs: Vrai si aucun passage n'a passé le seuil de pertinence
    """
    if no_relevant_docs:
        st.warning(
            "Aucun passage de la base n'est assez proche de la question : "
            "la réponse n'a pas été générée pour éviter une réponse inventée."
        )
        return
    if not passages:
        return

    scores = [p["score"] for p in passages if p.get("score") is not None]
    if scores and max(scores) < config.RELEVANCE_WARNING_SCORE:
        st.warning(
            "Les passages trouvés sont peu pertinents : vérifiez la réponse "
            "dans les documents."
        )
    if any(p.get("ocr") or p.get("ocr_images") for p in passages):
        st.caption("Certains passages proviennent de l'OCR : des erreurs de lecture sont possibles.")

    with st.expander(f"📚 Sources citées ({len(passages)})", expanded=False):
        tabs = st.tabs([_tab_label(p) for p in passages])
        for tab, passage in zip(tabs, passages):
            with tab:
                _render_passage(passage, f"{key_prefix}_{passage['num']}")
