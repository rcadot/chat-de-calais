# temp_documents.py
"""Gestion des documents temporaires pour une session utilisateur."""

import tempfile
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loaders import load_document
import config


def init_temp_session():
    """Initialise les variables de session pour les docs temporaires."""
    if "temp_documents" not in st.session_state:
        st.session_state.temp_documents = []
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    if "original_retriever" not in st.session_state:
        st.session_state.original_retriever = None


def save_uploaded_file(uploaded_file) -> Path:
    """
    Sauvegarde un fichier uploadé dans le dossier temporaire.

    Args:
        uploaded_file: Fichier Streamlit uploadé

    Returns:
        Path: Chemin vers le fichier sauvegardé
    """
    temp_path = Path(st.session_state.temp_dir) / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def load_and_store_document(uploaded_file) -> Dict[str, Any]:
    """
    Charge un document et le stocke en session.

    Args:
        uploaded_file: Fichier Streamlit uploadé

    Returns:
        Dict contenant les infos du document
    """
    # Sauvegarder le fichier
    temp_path = save_uploaded_file(uploaded_file)

    # Charger et découper
    docs = load_document(str(temp_path))

    # Créer l'entrée
    doc_entry = {
        "name": uploaded_file.name,
        "path": str(temp_path),
        "chunks": len(docs),
        "docs": docs,
        "size": uploaded_file.size,
    }

    return doc_entry


def create_temp_retriever(embeddings, permanent_docs: List[Document] = None):
    """
    Crée un retriever temporaire avec les docs permanents + temporaires.

    Args:
        embeddings: Modèle d'embeddings
        permanent_docs: Documents permanents (optionnel)

    Returns:
        Retriever combiné
    """
    # Collecter tous les documents temporaires
    all_temp_docs = []
    for temp_doc in st.session_state.temp_documents:
        all_temp_docs.extend(temp_doc["docs"])

    # Si pas de docs permanents fournis, utiliser seulement les temporaires
    if permanent_docs is None:
        all_docs = all_temp_docs
    else:
        all_docs = permanent_docs + all_temp_docs

    # Créer un vectorstore en mémoire
    if all_docs:
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            collection_name=f"session_{id(st.session_state)}",
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.RAG_TOP_N_RETRIEVAL}
        )

        return retriever

    return None


def add_temp_documents(uploaded_files, embeddings) -> tuple[int, int]:
    """
    Ajoute des documents temporaires et réindexe.

    Args:
        uploaded_files: Liste de fichiers uploadés
        embeddings: Modèle d'embeddings

    Returns:
        Tuple (nombre de nouveaux docs, nombre total de chunks)
    """
    # Sauvegarder le retriever original si première utilisation
    if st.session_state.original_retriever is None:
        st.session_state.original_retriever = st.session_state.retriever

    existing_names = {d["name"] for d in st.session_state.temp_documents}
    new_docs_count = 0
    total_chunks = 0

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in existing_names:
            try:
                doc_entry = load_and_store_document(uploaded_file)
                st.session_state.temp_documents.append(doc_entry)
                new_docs_count += 1
                total_chunks += doc_entry["chunks"]
            except Exception as e:
                st.error(f"❌ Erreur avec {uploaded_file.name}: {str(e)}")

    # Réindexer si de nouveaux docs
    if new_docs_count > 0:
        # Créer retriever temporaire (sans docs permanents pour éviter duplication)
        temp_retriever = create_temp_retriever(embeddings)

        if temp_retriever:
            st.session_state.retriever = temp_retriever

    return new_docs_count, total_chunks


def remove_temp_document(doc_name: str, embeddings):
    """
    Retire un document temporaire et réindexe.

    Args:
        doc_name: Nom du document à retirer
        embeddings: Modèle d'embeddings
    """
    # Retirer le document
    st.session_state.temp_documents = [
        d for d in st.session_state.temp_documents if d["name"] != doc_name
    ]

    # Réindexer
    if st.session_state.temp_documents:
        temp_retriever = create_temp_retriever(embeddings)
        if temp_retriever:
            st.session_state.retriever = temp_retriever
    else:
        # Restaurer le retriever original
        if st.session_state.original_retriever:
            st.session_state.retriever = st.session_state.original_retriever


def clear_all_temp_documents():
    """Efface tous les documents temporaires et restaure le retriever original."""
    st.session_state.temp_documents = []

    # Restaurer le retriever original
    if st.session_state.original_retriever:
        st.session_state.retriever = st.session_state.original_retriever
        st.session_state.original_retriever = None


def get_temp_docs_info() -> Dict[str, Any]:
    """
    Retourne les informations sur les documents temporaires.

    Returns:
        Dict avec statistiques
    """
    if not st.session_state.temp_documents:
        return {"count": 0, "total_chunks": 0, "total_size": 0}

    total_chunks = sum(d["chunks"] for d in st.session_state.temp_documents)
    total_size = sum(d["size"] for d in st.session_state.temp_documents)

    return {
        "count": len(st.session_state.temp_documents),
        "total_chunks": total_chunks,
        "total_size": total_size,
        "documents": st.session_state.temp_documents,
    }


def format_file_size(size_bytes: int) -> str:
    """Formate la taille d'un fichier en unités lisibles."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def render_temp_documents_section(embeddings):
    """
    Affiche la section complète de gestion des documents temporaires.
    À appeler dans la sidebar.

    Args:
        embeddings: Modèle d'embeddings
    """
    st.subheader("📤 Documents temporaires")

    # Initialiser
    init_temp_session()

    # File uploader
    uploaded_files = st.file_uploader(
        "Ajouter des documents",
        type=["pdf", "txt", "md", "docx", "doc", "odt", "html", "htm"],
        accept_multiple_files=True,
        help="Documents valables uniquement pour cette session",
        key="temp_doc_uploader",
        # label_visibility="collapsed",
    )

    # Traiter les uploads
    if uploaded_files:
        with st.spinner("📚 Chargement des documents..."):
            new_count, total_chunks = add_temp_documents(uploaded_files, embeddings)

            if new_count > 0:
                st.success(
                    f"✅ {new_count} document(s) ajouté(s) ({total_chunks} chunks)"
                )

    # Afficher les documents actifs
    info = get_temp_docs_info()

    if info["count"] > 0:
        st.write(f"**📊 {info['count']} document(s) actif(s)**")
        st.caption(
            f"Total: {info['total_chunks']} chunks • {format_file_size(info['total_size'])}"
        )

        # Liste des documents
        for i, doc in enumerate(info["documents"]):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {doc['name']}")
                st.caption(
                    f"   {doc['chunks']} chunks • {format_file_size(doc['size'])}"
                )
            with col2:
                if st.button("🗑️", key=f"remove_temp_{i}", help="Retirer"):
                    remove_temp_document(doc["name"], embeddings)
                    st.rerun()

        # Bouton tout effacer
        st.divider()
        if st.button("🗑️ Tout effacer", key="clear_all_temp", use_container_width=True):
            clear_all_temp_documents()
            st.success("✅ Documents temporaires effacés")
            st.rerun()
    else:
        st.info("Aucun document temporaire")
