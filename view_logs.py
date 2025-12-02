# view_logs.py
"""Visualisation CLI des logs RAG."""
import argparse
from datetime import datetime
from pathlib import Path
from logger import RAGLogger
import json


def format_timestamp(ts: str) -> str:
    """Formate le timestamp de manière lisible."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return ts


def view_recent(limit: int = 10):
    """Affiche les requêtes récentes."""
    logger = RAGLogger()
    queries = logger.get_recent_queries(limit)
    
    print(f"\n📜 {len(queries)} DERNIÈRES REQUÊTES\n")
    print("=" * 100)
    
    for i, q in enumerate(queries, 1):
        status = "✅" if not q['error'] else "❌"
        timestamp = format_timestamp(q['timestamp'])
        
        print(f"\n{status} #{q['id']} - {timestamp}")
        print(f"   Question: {q['user_query']}")
        
        if q['error']:
            print(f"   ❌ Erreur: {q['error']}")
        else:
            answer_preview = q['final_answer'][:150] + "..." if len(q['final_answer']) > 150 else q['final_answer']
            print(f"   Réponse: {answer_preview}")
            
            # Sources
            if q['sources']:
                try:
                    sources = json.loads(q['sources'])
                    source_names = [Path(s).name for s in sources[:3]]
                    print(f"   📁 Sources: {', '.join(source_names)}")
                except:
                    pass
            
            # Stats
            print(f"   📊 Docs: {q['retrieved_docs_count']} → {q['reranked_docs_count']}")
            if q['execution_time_seconds']:
                print(f"   ⏱️  Temps: {q['execution_time_seconds']:.2f}s")
        
        print("-" * 100)


def view_detail(query_id: int):
    """Affiche le détail complet d'une requête."""
    logger = RAGLogger()
    
    import sqlite3
    conn = sqlite3.connect(logger.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM rag_queries WHERE id = ?", (query_id,))
    q = cursor.fetchone()
    conn.close()
    
    if not q:
        print(f"❌ Requête #{query_id} introuvable")
        return
    
    q = dict(q)
    
    print(f"\n{'='*100}")
    print(f"DÉTAIL REQUÊTE #{q['id']}")
    print(f"{'='*100}\n")
    
    print(f"📅 Date: {format_timestamp(q['timestamp'])}")
    print(f"❓ Question: {q['user_query']}")
    
    if q['hyde_query']:
        print(f"\n🧠 HyDE Query:")
        print(f"   {q['hyde_query'][:300]}...")
    
    print(f"\n📊 Pipeline:")
    print(f"   Docs récupérés: {q['retrieved_docs_count']}")
    print(f"   Docs après rerank: {q['reranked_docs_count']}")
    if q['execution_time_seconds']:
        print(f"   Temps d'exécution: {q['execution_time_seconds']:.2f}s")
    
    if q['error']:
        print(f"\n❌ ERREUR:")
        print(f"   {q['error']}")
    else:
        print(f"\n📄 RÉPONSE:")
        print(f"   {q['final_answer']}\n")
        
        if q['sources']:
            try:
                sources = json.loads(q['sources'])
                print(f"📁 SOURCES ({len(sources)}):")
                for i, s in enumerate(sources, 1):
                    print(f"   {i}. {s}")
            except:
                pass
        
        if q['rerank_scores']:
            try:
                scores = json.loads(q['rerank_scores'])
                print(f"\n🎯 SCORES RERANK:")
                for i, score in enumerate(scores, 1):
                    if score is not None:
                        print(f"   {i}. {score:.4f}")
            except:
                pass
        
        if q['reranked_docs_details']:
            try:
                docs = json.loads(q['reranked_docs_details'])
                print(f"\n📝 DOCUMENTS UTILISÉS:")
                for i, doc in enumerate(docs, 1):
                    print(f"\n   {i}. {Path(doc['source']).name}")
                    if doc.get('rerank_score'):
                        print(f"      Score: {doc['rerank_score']:.4f}")
                    print(f"      Extrait: {doc['content'][:150]}...")
            except:
                pass


def view_stats():
    """Affiche les statistiques globales."""
    logger = RAGLogger()
    stats = logger.get_stats()
    
    print("\n📊 STATISTIQUES GLOBALES\n")
    print("=" * 60)
    
    print(f"\nRequêtes:")
    print(f"  Total: {stats['total_queries']}")
    print(f"  Succès: {stats['total_queries'] - stats['error_queries']}")
    print(f"  Erreurs: {stats['error_queries']}")
    print(f"  Taux de succès: {stats['success_rate']:.1f}%")
    
    if stats['avg_execution_time']:
        print(f"\nPerformances:")
        print(f"  Temps moyen: {stats['avg_execution_time']:.2f}s")
    
    if stats['avg_docs_retrieved']:
        print(f"\nDocuments:")
        print(f"  Docs moyens récupérés: {stats['avg_docs_retrieved']:.1f}")
    
    print("\n" + "=" * 60)


def search_logs(search_term: str, limit: int = 10):
    """Recherche dans les logs."""
    logger = RAGLogger()
    results = logger.search_queries(search_term, limit)
    
    print(f"\n🔍 Recherche: '{search_term}' - {len(results)} résultats\n")
    print("=" * 100)
    
    for q in results:
        timestamp = format_timestamp(q['timestamp'])
        status = "✅" if not q['error'] else "❌"
        
        print(f"\n{status} #{q['id']} - {timestamp}")
        print(f"   Question: {q['user_query']}")
        
        if not q['error']:
            answer = q['final_answer'][:100] + "..." if len(q['final_answer']) > 100 else q['final_answer']
            print(f"   Réponse: {answer}")
        
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Visualisation des logs RAG")
    subparsers = parser.add_subparsers(dest="command", help="Commandes")
    
    # Recent
    recent_parser = subparsers.add_parser("recent", help="Requêtes récentes")
    recent_parser.add_argument("--limit", type=int, default=10, help="Nombre de requêtes")
    
    # Detail
    detail_parser = subparsers.add_parser("detail", help="Détail d'une requête")
    detail_parser.add_argument("id", type=int, help="ID de la requête")
    
    # Stats
    subparsers.add_parser("stats", help="Statistiques globales")
    
    # Search
    search_parser = subparsers.add_parser("search", help="Rechercher dans les logs")
    search_parser.add_argument("term", type=str, help="Terme à rechercher")
    search_parser.add_argument("--limit", type=int, default=10, help="Nombre de résultats")
    
    args = parser.parse_args()
    
    if args.command == "recent":
        view_recent(args.limit)
    elif args.command == "detail":
        view_detail(args.id)
    elif args.command == "stats":
        view_stats()
    elif args.command == "search":
        search_logs(args.term, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
