# tests/test_config.py
"""Tests du module config."""
import pytest
import os

def test_config_loads_env_vars(monkeypatch):
    """Test que la config charge les variables d'environnement."""
    monkeypatch.setenv("ALBERT_API_KEY", "test_key")
    
    # Recharger config
    import importlib
    import config
    importlib.reload(config)
    
    assert config.ALBERT_API_KEY == "test_key"

def test_config_has_required_constants():
    """Test que toutes les constantes requises sont définies."""
    import config
    
    required = [
        'ALBERT_API_KEY',
        'ALBERT_BASE_URL',
        'EMBEDDINGS_MODEL',
        'LLM_MODEL',
        'RERANK_MODEL',
        'CHUNK_SIZE',
        'CHUNK_OVERLAP',
        'DOCUMENTS_DIR',
        'CHROMA_DB_PATH',
        'RAG_LOGS_DB',
    ]
    
    for const in required:
        assert hasattr(config, const), f"Constante manquante: {const}"

def test_config_chunk_parameters_are_valid():
    """Test que les paramètres de chunking sont valides."""
    import config
    
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert isinstance(config.CHUNK_SEPARATORS, list)
