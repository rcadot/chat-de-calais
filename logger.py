# logger.py (VERSION MISE À JOUR)
"""Logger pour les requêtes RAG."""
import sqlite3
from datetime import datetime
import json
from typing import List, Optional

class RAGLogger:
    """Logger SQLite pour les requêtes RAG."""
    
    def __init__(self, db_path: str = "rag_logs.db"):
        """
        Initialise le logger.
        
        Args:
            db_path: Chemin vers la base SQLite
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Crée la table si elle n'existe pas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_query TEXT NOT NULL,
                hyde_query TEXT,
                retrieved_docs_count INTEGER,
                reranked_docs_count INTEGER,
                final_answer TEXT,
                sources TEXT,
                rerank_scores TEXT,
                execution_time_seconds REAL,
                error TEXT,
                retrieved_docs_details TEXT,
                reranked_docs_details TEXT,
                prompt_mode TEXT  -- ✅ NOUVELLE COLONNE
            )
        """)
        
        # ✅ Ajouter la colonne si elle n'existe pas (migration)
        try:
            cursor.execute("ALTER TABLE rag_queries ADD COLUMN prompt_mode TEXT")
        except sqlite3.OperationalError:
            # Colonne existe déjà
            pass
        
        conn.commit()
        conn.close()
    
    def log_query(
        self,
        user_query: str,
        hyde_query: Optional[str] = None,
        retrieved_docs: Optional[List] = None,
        reranked_docs: Optional[List] = None,
        final_answer: Optional[str] = None,
        sources: Optional[List[str]] = None,
        rerank_scores: Optional[List[float]] = None,
        execution_time: Optional[float] = None,
        error: Optional[str] = None,
        prompt_mode: Optional[str] = None  # ✅ NOUVEAU PARAMÈTRE
    ) -> int:
        """
        Log une requête RAG.
        
        Args:
            user_query: Question de l'utilisateur
            hyde_query: Query après HyDE
            retrieved_docs: Documents récupérés
            reranked_docs: Documents après rerank
            final_answer: Réponse générée
            sources: Liste des sources utilisées
            rerank_scores: Scores de rerank
            execution_time: Temps d'exécution en secondes
            error: Message d'erreur éventuel
            prompt_mode: Mode de prompt utilisé (administratif, technique, créatif)
        
        Returns:
            int: ID de la requête loggée
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Préparer les données
        timestamp = datetime.now().isoformat()
        
        retrieved_count = len(retrieved_docs) if retrieved_docs else 0
        reranked_count = len(reranked_docs) if reranked_docs else 0
        
        sources_json = json.dumps(sources) if sources else None
        scores_json = json.dumps(rerank_scores) if rerank_scores else None
        
        # Détails des docs (optionnel)
        retrieved_details = None
        if retrieved_docs:
            retrieved_details = json.dumps([
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "content": doc.page_content[:200]
                }
                for doc in retrieved_docs[:10]  # Limiter à 10
            ])
        
        reranked_details = None
        if reranked_docs:
            reranked_details = json.dumps([
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "content": doc.page_content[:200],
                    "rerank_score": doc.metadata.get("rerank_score")
                }
                for doc in reranked_docs
            ])
        
        cursor.execute("""
            INSERT INTO rag_queries (
                timestamp, user_query, hyde_query,
                retrieved_docs_count, reranked_docs_count,
                final_answer, sources, rerank_scores,
                execution_time_seconds, error,
                retrieved_docs_details, reranked_docs_details,
                prompt_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, user_query, hyde_query,
            retrieved_count, reranked_count,
            final_answer, sources_json, scores_json,
            execution_time, error,
            retrieved_details, reranked_details,
            prompt_mode  # ✅ AJOUTER le mode
        ))
        
        query_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return query_id
    
    def get_recent_queries(self, limit: int = 10) -> List[dict]:
        """Récupère les dernières requêtes."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM rag_queries 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_stats(self) -> dict:
        """Calcule les statistiques globales."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total requêtes
        cursor.execute("SELECT COUNT(*) FROM rag_queries")
        total_queries = cursor.fetchone()[0]
        
        # Erreurs
        cursor.execute("SELECT COUNT(*) FROM rag_queries WHERE error IS NOT NULL")
        error_queries = cursor.fetchone()[0]
        
        # Temps moyen
        cursor.execute("SELECT AVG(execution_time_seconds) FROM rag_queries WHERE error IS NULL")
        avg_time = cursor.fetchone()[0] or 0
        
        # Docs moyens
        cursor.execute("SELECT AVG(retrieved_docs_count) FROM rag_queries WHERE error IS NULL")
        avg_docs = cursor.fetchone()[0] or 0
        
        # ✅ Stats par mode
        cursor.execute("""
            SELECT prompt_mode, COUNT(*) as count
            FROM rag_queries
            WHERE prompt_mode IS NOT NULL
            GROUP BY prompt_mode
        """)
        mode_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_queries": total_queries,
            "error_queries": error_queries,
            "success_rate": ((total_queries - error_queries) / total_queries * 100) if total_queries > 0 else 0,
            "avg_execution_time": avg_time,
            "avg_docs_retrieved": avg_docs,
            "mode_distribution": mode_stats  # ✅ AJOUTER distribution des modes
        }
    
    def search_queries(self, search_term: str, limit: int = 10) -> List[dict]:
        """Recherche dans les requêtes."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM rag_queries 
            WHERE user_query LIKE ? OR final_answer LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{search_term}%", f"%{search_term}%", limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
