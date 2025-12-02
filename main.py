# main.py (version streaming pour console)
import sys
from albert_client import get_embeddings, get_llm
from indexer import index_documents
from rag_pipeline import rag_query_stream  # ✅ Importer version stream
from logger import RAGLogger

def main():
    """Fonction principale avec streaming."""
    
    print("🚀 Initialisation du système RAG...\n")
    
    # Init
    embeddings = get_embeddings()
    llm = get_llm()
    logger = RAGLogger()
    
    # Indexer
    vectorstore, retriever = index_documents(embeddings)
    
    # Mode interactif
    print("\n💬 Mode interactif avec streaming (tapez 'quit' pour quitter)\n")
    
    while True:
        query = input("❓ Question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Au revoir !")
            break
        
        if not query:
            continue
        
        print(f"\n📄 Réponse:\n")
        
        # ✅ STREAMING : affiche caractère par caractère
        try:
            for chunk in rag_query_stream(query, retriever, llm, logger):
                print(chunk, end='', flush=True)  # flush=True pour affichage immédiat
            
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ Erreur: {e}\n")


if __name__ == "__main__":
    main()
