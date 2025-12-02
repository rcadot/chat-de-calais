# tests/conftest.py
"""Fixtures partagées pour tous les tests."""
import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime

@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)

@pytest.fixture
def temp_documents_dir(temp_dir):
    """Crée un répertoire de documents temporaire avec des fichiers de test."""
    docs_dir = os.path.join(temp_dir, "documents")
    os.makedirs(docs_dir)
    
    # Créer fichiers de test
    # TXT
    with open(os.path.join(docs_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write("Ceci est un document de test sur le ZAN.\nDeuxième ligne.")
    
    # MD
    with open(os.path.join(docs_dir, "test.md"), "w", encoding="utf-8") as f:
        f.write("# Titre\n\nContenu markdown sur Mad et Moselle.")
    
    return docs_dir

@pytest.fixture
def mock_embeddings():
    """Mock des embeddings Albert."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 5
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock

# tests/conftest.py (VERSION ULTRA-ROBUSTE)

@pytest.fixture
def mock_llm():
    """Mock du LLM Albert - version ultra-robuste."""
    from unittest.mock import Mock, MagicMock
    
    # ✅ Mock principal
    mock = Mock()
    
    # ✅ Classe pour simuler AIMessage
    class FakeAIMessage:
        def __init__(self, content="Réponse du LLM de test."):
            self.content = content
            
        def __str__(self):
            return self.content
    
    # ✅ invoke() retourne AIMessage
    mock.invoke.return_value = FakeAIMessage()
    
    # ✅ stream() retourne des chunks
    def fake_stream(*args, **kwargs):
        for chunk in ["Chunk ", "de ", "test"]:
            yield FakeAIMessage(chunk)
    
    mock.stream = fake_stream
    
    # ✅ __or__ pour supporter prompt | llm | parser
    original_mock = mock
    
    def create_chain(other):
        """Crée un chain qui fonctionne."""
        chain = Mock()
        
        # Selon ce qui est "pipé" avec le LLM
        if hasattr(other, '__name__'):
            if 'Parser' in str(other):
                # Si c'est un parser, invoke retourne string
                chain.invoke.return_value = "Réponse parsée en string."
            else:
                # Sinon retourne AIMessage
                chain.invoke.return_value = FakeAIMessage()
        else:
            # Par défaut, comportement du LLM original
            chain.invoke = original_mock.invoke
            chain.stream = original_mock.stream
        
        # Support du chaînage multiple (prompt | llm | parser)
        chain.__or__ = create_chain
        
        return chain
    
    mock.__or__ = create_chain
    
    return mock


@pytest.fixture
def sample_documents():
    """Documents LangChain de test."""
    from langchain_core.documents import Document
    return [
        Document(page_content="Contenu 1 sur le ZAN", metadata={"source": "doc1.pdf"}),
        Document(page_content="Contenu 2 sur Mad et Moselle", metadata={"source": "doc2.odt"}),
        Document(page_content="Contenu 3 général", metadata={"source": "doc3.txt"}),
    ]

@pytest.fixture
def mock_retriever(sample_documents):
    """Mock du retriever."""
    mock = MagicMock()
    mock.invoke.return_value = sample_documents
    return mock

@pytest.fixture
def temp_db_path(temp_dir):
    """Chemin vers une base ChromaDB temporaire."""
    return os.path.join(temp_dir, "test_chroma_db")

@pytest.fixture
def temp_logs_db(temp_dir):
    """Chemin vers une base de logs temporaire."""
    return os.path.join(temp_dir, "test_logs.db")

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, temp_dir):
    """Configure l'environnement pour les tests."""
    monkeypatch.setenv("ALBERT_API_KEY", "test_key_12345")
    
    # Importer config après avoir set les env vars
    import config
    monkeypatch.setattr(config, "VERBOSE", False)  # Désactiver logs pendant tests
