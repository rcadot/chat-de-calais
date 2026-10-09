# indexer.py
"""Indexation incrémentale des documents."""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
import corpus
from loaders import load_document
from retrieval import make_retriever

log = logging.getLogger(__name__)


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter commun à l'indexation permanente et aux documents temporaires."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    )


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Découpe les pages d'un fichier en chunks numérotés et contextualisés.

    - `chunk_index` (ordre de lecture dans le fichier) permet de retrouver les
      chunks voisins pour élargir le contexte d'un passage ;
    - les métadonnées de corpus (date de réunion, statut validé, titre,
      famille) sont tirées du chemin `source` ;
    - un en-tête contextuel (titre, date, statut) est placé devant le texte
      pour distinguer des passages proches issus de versions différentes ;
      `header_len` permet de le retirer à l'affichage (corpus.chunk_body).

    Args:
        docs: Pages (ou documents) d'un même fichier, dans l'ordre

    Returns:
        List[Document]: Chunks avec métadonnées chunk_index, header_len et corpus
    """
    chunks = get_text_splitter().split_documents(docs) if docs else []
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(corpus.document_metadata(chunk.metadata.get("source", "")))
        chunk.metadata["chunk_index"] = i
        header = corpus.context_header(chunk.metadata)
        chunk.page_content = header + chunk.page_content
        chunk.metadata["header_len"] = len(header)
    return chunks


def get_file_hash(file_path: str) -> str:
    """Hash MD5 du fichier (chaîne vide si illisible)."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except OSError as e:
        log.warning("Lecture impossible de %s : %s", file_path, e)
        return ""


def scan_documents() -> Dict[str, Dict]:
    """Scanne le dossier documents et retourne les métadonnées."""
    files = {}
    for root, dirs, filenames in os.walk(config.DOCUMENTS_DIR):
        for filename in filenames:
            if Path(filename).suffix.lower() in config.SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, filename)
                if corpus.is_excluded(file_path):
                    continue
                stat = os.stat(file_path)
                files[file_path] = {
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "hash": get_file_hash(file_path),
                }
    return files


def detect_changes(
    old_files: Dict, current_files: Dict
) -> Tuple[List, List, List, List]:
    """Détecte nouveaux, modifiés, supprimés, inchangés."""
    new = [f for f in current_files if f not in old_files]
    modified = [
        f
        for f in current_files
        if f in old_files and current_files[f]["hash"] != old_files[f].get("hash")
    ]
    deleted = [f for f in old_files if f not in current_files]
    unchanged = [
        f
        for f in current_files
        if f in old_files and current_files[f]["hash"] == old_files[f].get("hash")
    ]

    return new, modified, deleted, unchanged


def index_documents(embeddings):
    """
    Indexation incrémentale.

    Args:
        embeddings: Client embeddings

    Returns:
        (vectorstore, retriever)
    """
    db_path = os.path.abspath(config.CHROMA_DB_PATH)
    os.makedirs(db_path, exist_ok=True)

    log.info("Base: %s", db_path)
    log.info("Documents: %s\n", config.DOCUMENTS_DIR)

    # Charger métadonnées
    metadata_file = os.path.join(db_path, "index_metadata.json")
    old_files = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file) as f:
                old_files = json.load(f).get("files", {})
        except (OSError, ValueError) as e:
            log.warning("Métadonnées d'index illisibles (%s) : réindexation complète", e)

    # Scanner documents actuels
    current_files = scan_documents()
    new, modified, deleted, unchanged = detect_changes(old_files, current_files)

    log.info("Changements:")
    log.info("   Nouveaux: %d", len(new))
    log.info("   Modifiés: %d", len(modified))
    log.info("   Supprimés: %d", len(deleted))
    log.info("   Inchangés: %d", len(unchanged))

    client = chromadb.PersistentClient(
        path=db_path, settings=Settings(anonymized_telemetry=False)
    )
    vectorstore = Chroma(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Si aucun changement
    if not (new or modified or deleted):
        log.info("\nAucun changement!")
        return vectorstore, make_retriever(vectorstore)

    log.info("\nMise à jour...\n")
    collection = client.get_collection(name=config.COLLECTION_NAME)

    # Supprimer les chunks existants des fichiers disparus, modifiés, et aussi
    # nouveaux (reliquats d'une indexation précédente interrompue)
    for file_path in deleted + modified + new:
        try:
            results = collection.get(where={"source": file_path})
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                log.info("🗑️ %s: %d chunks supprimés", Path(file_path).name, len(results["ids"]))
        except Exception as e:
            log.warning("Erreur suppression %s: %s", Path(file_path).name, e)

    # Indexer nouveaux et modifiés
    files_to_index = new + modified
    failed = set()
    if files_to_index:
        log.info("\n Indexation de %d fichiers...\n", len(files_to_index))

        all_chunks = []
        for file_path in files_to_index:
            docs = load_document(file_path)
            chunks = split_documents(docs)
            if chunks:
                all_chunks.extend(chunks)
                n_ocr = sum(1 for d in docs if d.metadata.get("ocr"))
                ocr_info = f" (dont {n_ocr} page(s) OCR)" if n_ocr else ""
                log.info(" %s: %d chunks%s", Path(file_path).name, len(chunks), ocr_info)
            else:
                failed.add(file_path)

        # Batch indexing
        log.info("\n⏳ Ajout de %d chunks...", len(all_chunks))
        for i in range(0, len(all_chunks), config.BATCH_SIZE):
            batch = all_chunks[i : i + config.BATCH_SIZE]
            try:
                vectorstore.add_documents(batch)
            except Exception as e:
                # Fichiers du lot incomplets : à retenter au prochain lancement
                sources = {c.metadata.get("source") for c in batch}
                failed.update(sources)
                log.error(" Batch %d: %s (%d fichier(s) à retenter)", i, e, len(sources))

    # Les fichiers sans contenu extrait ou en échec d'ajout ne sont pas
    # enregistrés : ils seront retentés à la prochaine indexation.
    if failed:
        log.warning("\n⚠️ %d fichier(s) non indexé(s), à retenter :", len(failed))
        for file_path in sorted(failed):
            log.warning("   - %s", Path(file_path).name)
            current_files.pop(file_path, None)

    # Sauvegarder métadonnées
    with open(metadata_file, "w") as f:
        json.dump(
            {"files": current_files, "last_update": datetime.now().isoformat()},
            f,
            indent=2,
        )

    total_chunks = collection.count()
    log.info("\n Terminé! %d fichiers, %d chunks", len(current_files), total_chunks)

    return vectorstore, make_retriever(vectorstore)
