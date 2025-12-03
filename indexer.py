# indexer.py
"""Indexation incrémentale des documents."""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config
from loaders import load_document


def get_file_hash(file_path: str) -> str:
    """Hash MD5 du fichier."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return ""


def scan_documents() -> Dict[str, Dict]:
    """Scanne le dossier documents et retourne les métadonnées."""
    files = {}
    for root, dirs, filenames in os.walk(config.DOCUMENTS_DIR):
        for filename in filenames:
            if Path(filename).suffix.lower() in config.SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, filename)
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

    print(f"Base: {db_path}")
    print(f"Documents: {config.DOCUMENTS_DIR}\n")

    # Charger métadonnées
    metadata_file = os.path.join(db_path, "index_metadata.json")
    old_files = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file) as f:
                old_files = json.load(f).get("files", {})
        except:
            pass

    # Scanner documents actuels
    current_files = scan_documents()
    new, modified, deleted, unchanged = detect_changes(old_files, current_files)

    print("Changements:")
    print(f"   Nouveaux: {len(new)}")
    print(f"   Modifiés: {len(modified)}")
    print(f"   Supprimés: {len(deleted)}")
    print(f"   Inchangés: {len(unchanged)}")

    # Si aucun changement
    if not (new or modified or deleted):
        print("\nAucun changement!")
        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )
        vectorstore = Chroma(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVER_TOP_K}
        )
        return vectorstore, retriever

    print("\nMise à jour...\n")

    # Init vectorstore
    client = chromadb.PersistentClient(
        path=db_path, settings=Settings(anonymized_telemetry=False)
    )
    try:
        vectorstore = Chroma(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
        )
        print("Collection existante")
    except:
        vectorstore = Chroma(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
        )
        print("Nouvelle collection")

    collection = client.get_collection(name=config.COLLECTION_NAME)

    # Supprimer fichiers disparus ou modifiés
    for file_path in deleted + modified:
        try:
            results = collection.get(where={"source": file_path})
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                if config.VERBOSE:
                    print(
                        f"🗑️ {Path(file_path).name}: {len(results['ids'])} chunks supprimés"
                    )
        except Exception as e:
            if config.VERBOSE:
                print(f" Erreur suppression {Path(file_path).name}: {e}")

    # Indexer nouveaux et modifiés
    files_to_index = new + modified
    if files_to_index:
        print(f"\n Indexation de {len(files_to_index)} fichiers...\n")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.CHUNK_SEPARATORS,
        )

        all_chunks = []
        for file_path in files_to_index:
            docs = load_document(file_path)
            if docs:
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                if config.VERBOSE:
                    print(f" {Path(file_path).name}: {len(chunks)} chunks")

        # Batch indexing
        print(f"\n⏳ Ajout de {len(all_chunks)} chunks...")
        for i in range(0, len(all_chunks), config.BATCH_SIZE):
            batch = all_chunks[i : i + config.BATCH_SIZE]
            try:
                vectorstore.add_documents(batch)
            except Exception as e:
                print(f" Batch {i}: {e}")

    # Sauvegarder métadonnées
    with open(metadata_file, "w") as f:
        json.dump(
            {"files": current_files, "last_update": datetime.now().isoformat()},
            f,
            indent=2,
        )

    total_chunks = collection.count()
    print(f"\n Terminé! {len(current_files)} fichiers, {total_chunks} chunks")

    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})
    return vectorstore, retriever
