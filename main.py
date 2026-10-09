# main.py (version streaming pour console)
"""Indexation des documents puis chat en console (réponses en flux)."""
import config
from albert_client import get_embeddings, get_llm
from indexer import index_documents
from rag_pipeline import rag_query_stream
from logger import RAGLogger


def main():
    """Indexe les documents puis ouvre une session de questions en console."""
    config.setup_logging()
    print("🚀 Initialisation du système RAG...\n")

    # Init
    embeddings = get_embeddings()
    llm = get_llm()
    logger = RAGLogger()

    # Indexer
    vectorstore, retriever = index_documents(embeddings)

    # Mode interactif
    print("\n💬 Mode interactif avec streaming (tapez 'quit' pour quitter)\n")
    history = []

    while True:
        query = input("❓ Question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Au revoir !")
            break

        if not query:
            continue

        print()
        answer = ""
        for item in rag_query_stream(query, retriever, llm, logger, history=history):
            if item["type"] == "status":
                if item.get("detail"):
                    print(f"   ✓ {item['detail']} ({item['elapsed']:.1f} s)")
                else:
                    print(f"⏳ {item['content']}")
            elif item["type"] == "chunk":
                if not answer:
                    print("\n📄 Réponse:\n")
                answer += item["content"]
                print(item["content"], end="", flush=True)
            elif item["type"] == "error":
                print(f"\n❌ {item['content']}")
            elif item["type"] == "metadata":
                if item["passages"]:
                    print("\n\n📚 Sources :")
                    for p in item["passages"]:
                        score = f"{p['score']:.2f}" if p["score"] is not None else "N/A"
                        print(f"   [{p['num']}] {p['label']} (pertinence {score})")
                logger.log_query(
                    user_query=query,
                    hyde_query=item["hyde_query"],
                    retrieved_docs=item["retrieved_docs"],
                    reranked_docs=item["reranked_docs"],
                    final_answer=answer,
                    sources=item["sources"],
                    rerank_scores=item["rerank_scores"],
                    execution_time=item["execution_time"],
                    error=item["error"],
                    prompt_mode=item["prompt_mode"],
                )

        history += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
