# temp_documents.py
"""Gestion des documents temporaires pour une session utilisateur."""

import tempfile
import uuid
from itertools import zip_longest
from pathlib import Path
from typing import List, Dict, Any, Tuple
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from indexer import split_documents
from loaders import load_document
from retrieval import make_retriever
import config


class CombinedRetriever:
    """Interroge plusieurs retrievers et entrelace leurs résultats par rang."""

    supports_keyword_query = True

    def __init__(self, retrievers: List[Any]):
        """
        Args:
            retrievers: Retrievers à combiner (les None sont ignorés)
        """
        self.retrievers = [r for r in retrievers if r is not None]

    def invoke(
        self, query: str, keyword_query: str = None, filters: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Retourne les documents de chaque retriever, entrelacés par rang.

        Les filtres de corpus ne s'appliquent qu'au premier retriever (base
        permanente) : les documents temporaires sont toujours interrogés.
        """
        results = []
        for position, r in enumerate(self.retrievers):
            if getattr(r, "supports_keyword_query", False) is True:
                kwargs = {"keyword_query": keyword_query}
                if position == 0 and filters:
                    kwargs["filters"] = filters
                results.append(r.invoke(query, **kwargs))
            else:
                results.append(r.invoke(query))
        merged = []
        for group in zip_longest(*results):
            merged.extend(d for d in group if d is not None)
        return merged

    def get_neighbors(
        self, doc: Document, window: int
    ) -> Tuple[List[Document], List[Document]]:
        """Voisins du chunk, cherchés dans le premier retriever qui le connaît."""
        for retriever in self.retrievers:
            get_neighbors = getattr(retriever, "get_neighbors", None)
            if callable(get_neighbors):
                before, after = get_neighbors(doc, window)
                if before or after:
                    return before, after
        return [], []


def init_temp_session():
    """Initialise les variables de session pour les docs temporaires."""
    if "temp_documents" not in st.session_state:
        st.session_state.temp_documents = []
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    if "original_retriever" not in st.session_state:
        st.session_state.original_retriever = None
    if "temp_vectorstore" not in st.session_state:
        st.session_state.temp_vectorstore = None
    if "temp_retriever" not in st.session_state:
        st.session_state.temp_retriever = None
    if "temp_only" not in st.session_state:
        st.session_state.temp_only = False


def apply_session_retriever():
    """
    Positionne le retriever de session : base permanente et documents
    temporaires combinés, ou documents temporaires seuls si demandé.
    """
    temp_retriever = st.session_state.temp_retriever
    original = st.session_state.original_retriever
    if temp_retriever is None:
        if original is not None:
            st.session_state.retriever = original
        return
    if st.session_state.temp_only or original is None:
        st.session_state.retriever = temp_retriever
    else:
        st.session_state.retriever = CombinedRetriever([original, temp_retriever])


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

    # Charger et découper (même découpage que l'indexation permanente)
    docs = split_documents(load_document(str(temp_path)))

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

    # Supprimer la collection précédente : sinon les clients Chroma éphémères
    # partagent la collection et les documents s'y accumulent en double.
    previous = st.session_state.get("temp_vectorstore")
    if previous is not None:
        try:
            previous.delete_collection()
        except Exception:
            pass
        st.session_state.temp_vectorstore = None

    # Créer un vectorstore en mémoire
    if all_docs:
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            collection_name=f"session_{uuid.uuid4().hex[:12]}",
        )
        st.session_state.temp_vectorstore = vectorstore

        return make_retriever(vectorstore, docs=all_docs)

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

    # Réindexer si de nouveaux docs (les docs permanents restent interrogés
    # via leur propre retriever, sans être réindexés)
    if new_docs_count > 0:
        st.session_state.temp_retriever = create_temp_retriever(embeddings)
        apply_session_retriever()

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

    # Réindexer (create_temp_retriever renvoie None s'il ne reste rien,
    # ce qui restaure le retriever original)
    st.session_state.temp_retriever = create_temp_retriever(embeddings)
    apply_session_retriever()


def clear_all_temp_documents():
    """Efface tous les documents temporaires et restaure le retriever original."""
    st.session_state.temp_documents = []
    st.session_state.temp_retriever = None
    previous = st.session_state.get("temp_vectorstore")
    if previous is not None:
        try:
            previous.delete_collection()
        except Exception:
            pass
        st.session_state.temp_vectorstore = None

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
        type=[ext.lstrip(".") for ext in config.SUPPORTED_EXTENSIONS],
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
        st.checkbox(
            "Interroger uniquement ces documents",
            key="temp_only",
            help="Sinon, la base permanente est aussi interrogée",
        )
        apply_session_retriever()

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
