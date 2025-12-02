# tests/test_indexer.py
"""Tests du module indexer."""
import pytest
import os
import json
from indexer import (
    get_file_hash,
    scan_documents,
    detect_changes,
    index_documents
)

def test_get_file_hash(temp_documents_dir):
    """Test calcul hash MD5."""
    file_path = os.path.join(temp_documents_dir, "test.txt")
    hash1 = get_file_hash(file_path)
    
    assert hash1 != ""
    assert len(hash1) == 32  # MD5 hash
    
    # Hash identique pour même contenu
    hash2 = get_file_hash(file_path)
    assert hash1 == hash2

def test_scan_documents(temp_documents_dir):
    """Test scan du dossier documents."""
    import config
    import importlib
    
    # Override config pour tests
    config.DOCUMENTS_DIR = temp_documents_dir
    config.SUPPORTED_EXTENSIONS = ['.txt', '.md']
    importlib.reload(config)
    
    files = scan_documents()
    
    assert len(files) >= 2
    assert all("hash" in meta for meta in files.values())
    assert all("size" in meta for meta in files.values())

def test_detect_changes_new_files():
    """Test détection nouveaux fichiers."""
    old_files = {"file1.txt": {"hash": "abc123"}}
    current_files = {
        "file1.txt": {"hash": "abc123"},
        "file2.txt": {"hash": "def456"}
    }
    
    new, modified, deleted, unchanged = detect_changes(old_files, current_files)
    
    assert len(new) == 1
    assert "file2.txt" in new
    assert len(modified) == 0
    assert len(deleted) == 0
    assert len(unchanged) == 1

def test_detect_changes_modified_files():
    """Test détection fichiers modifiés."""
    old_files = {"file1.txt": {"hash": "abc123"}}
    current_files = {"file1.txt": {"hash": "xyz789"}}
    
    new, modified, deleted, unchanged = detect_changes(old_files, current_files)
    
    assert len(new) == 0
    assert len(modified) == 1
    assert "file1.txt" in modified
    assert len(deleted) == 0

def test_detect_changes_deleted_files():
    """Test détection fichiers supprimés."""
    old_files = {
        "file1.txt": {"hash": "abc123"},
        "file2.txt": {"hash": "def456"}
    }
    current_files = {"file1.txt": {"hash": "abc123"}}
    
    new, modified, deleted, unchanged = detect_changes(old_files, current_files)
    
    assert len(new) == 0
    assert len(modified) == 0
    assert len(deleted) == 1
    assert "file2.txt" in deleted

def test_index_documents_creates_vectorstore(temp_documents_dir, temp_db_path, mock_embeddings, monkeypatch):
    """Test création du vectorstore."""
    import config
    monkeypatch.setattr(config, "DOCUMENTS_DIR", temp_documents_dir)
    monkeypatch.setattr(config, "CHROMA_DB_PATH", temp_db_path)
    
    vectorstore, retriever = index_documents(mock_embeddings)
    
    assert vectorstore is not None
    assert retriever is not None
    assert os.path.exists(temp_db_path)
