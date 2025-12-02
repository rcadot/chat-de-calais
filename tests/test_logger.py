# tests/test_logger.py
"""Tests du module logger."""
import pytest
from logger import RAGLogger
from datetime import datetime

def test_logger_creates_database(temp_logs_db):
    """Test création de la base de données."""
    logger = RAGLogger(temp_logs_db)
    
    import sqlite3
    conn = sqlite3.connect(temp_logs_db)
    cursor = conn.cursor()
    
    # Vérifier que la table existe
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_queries'")
    assert cursor.fetchone() is not None
    
    conn.close()

def test_log_query_success(temp_logs_db):
    """Test log d'une requête réussie."""
    logger = RAGLogger(temp_logs_db)
    
    query_id = logger.log_query(
        user_query="Test question",
        hyde_query="Test hyde",
        retrieved_docs=None,
        reranked_docs=None,
        final_answer="Test answer",
        sources=["doc1.pdf", "doc2.txt"],
        rerank_scores=[0.9, 0.7],
        execution_time=1.5,
        error=None
    )
    
    assert query_id > 0
    
    # Vérifier données
    queries = logger.get_recent_queries(1)
    assert len(queries) == 1
    assert queries[0]["user_query"] == "Test question"
    assert queries[0]["final_answer"] == "Test answer"

def test_log_query_error(temp_logs_db):
    """Test log d'une requête avec erreur."""
    logger = RAGLogger(temp_logs_db)
    
    query_id = logger.log_query(
        user_query="Failed question",
        error="Test error message",
        execution_time=0.5
    )
    
    assert query_id > 0
    
    queries = logger.get_recent_queries(1)
    assert queries[0]["error"] == "Test error message"

def test_get_stats(temp_logs_db):
    """Test statistiques."""
    logger = RAGLogger(temp_logs_db)
    
    # Log quelques requêtes
    for i in range(5):
        logger.log_query(
            user_query=f"Question {i}",
            final_answer=f"Answer {i}",
            execution_time=1.0 + i * 0.1
        )
    
    # Log une erreur
    logger.log_query(
        user_query="Error query",
        error="Test error",
        execution_time=0.5
    )
    
    stats = logger.get_stats()
    
    assert stats["total_queries"] == 6
    assert stats["error_queries"] == 1
    assert stats["success_rate"] > 80

def test_search_queries(temp_logs_db):
    """Test recherche dans les logs."""
    logger = RAGLogger(temp_logs_db)
    
    logger.log_query(user_query="Question about ZAN", final_answer="Answer about ZAN")
    logger.log_query(user_query="Question about Mad", final_answer="Answer about Mad")
    logger.log_query(user_query="Other question", final_answer="Other answer")
    
    results = logger.search_queries("ZAN", limit=10)
    
    assert len(results) == 1
    assert "ZAN" in results[0]["user_query"]
