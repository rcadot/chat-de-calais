# rag_pipeline.py (VERSION COMPLÈTE)
"""Pipeline RAG avec HyDE et Reranking."""

import time
import requests
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

import sys
import io

# Forcer l'encodage UTF-8 pour éviter les erreurs charmap sous Windows
# if sys.platform == "win32":
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def generate_hyde(query: str, llm) -> str:
    """
    Génère un document hypothétique avec HyDE en utilisant le prompt du mode configuré.

    Args:
        query: Question de l'utilisateur
        llm: Modèle LLM

    Returns:
        Query enrichie avec document hypothétique
    """
    if not config.USE_HYDE:
        return query

    try:
        # Utiliser le template HyDE du mode configuré
        hyde_template = config.get_prompt_template("hyde")
        hyde_prompt = PromptTemplate.from_template(hyde_template)

        chain = hyde_prompt | llm | StrOutputParser()
        hyde_doc = chain.invoke({"query": query})

        # Combiner query originale + document hypothétique
        enhanced_query = f"{query}\n\n{hyde_doc}"

        if config.VERBOSE:
            print(f"🧠 HyDE (mode: {config.PROMPT_MODE}):")
            print(f"   Original: {query[:100]}...")
            print(f"   Enrichi: {hyde_doc[:200]}...")

        return enhanced_query

    except Exception as e:
        if config.VERBOSE:
            print(f"⚠️ Erreur HyDE: {e}")
        return query


def rerank_documents(query: str, docs: List, top_k: int = None) -> List:
    """
    Rerank les documents avec Albert API et ajoute les scores aux métadonnées.

    Args:
        query: Question de l'utilisateur
        docs: Liste de documents à reranker
        top_k: Nombre de documents à retourner

    Returns:
        Liste de documents reranked avec scores dans metadata
    """
    if not config.USE_RERANK or not docs:
        return docs[:top_k] if top_k else docs

    top_k = top_k or config.RAG_TOP_K_DOCS

    try:
        # Préparer les documents (limiter la longueur)
        doc_texts = [doc.page_content[:1000] for doc in docs]

        payload = {
            "model": config.RERANK_MODEL,
            "prompt": query,  # ✅ Format Albert API
            "input": doc_texts,
        }

        headers = {
            "Authorization": f"Bearer {config.ALBERT_API_KEY}",
            "Content-Type": "application/json",
        }

        if config.VERBOSE:
            print(f"🔄 Rerank: {len(doc_texts)} docs...")

        response = requests.post(
            f"{config.ALBERT_BASE_URL}/rerank",
            json=payload,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            results = response.json()

            if config.VERBOSE:
                print(f"✅ Rerank OK")

            # Parser et stocker les scores
            reranked_docs = []

            # Adapter selon le format de réponse d'Albert
            if isinstance(results, list):
                # Format: [{"index": 0, "score": 0.9}, ...]
                for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True)[
                    :top_k
                ]:
                    doc = docs[r["index"]]
                    # Ajouter le score dans les métadonnées
                    doc.metadata["rerank_score"] = r.get("score", 0.0)
                    reranked_docs.append(doc)

            elif isinstance(results, dict):
                result_list = results.get("results", results.get("data", []))

                for r in result_list[:top_k]:
                    idx = r.get("index")
                    score = r.get(
                        "score", r.get("relevance_score", r.get("rerank_score", 0.0))
                    )

                    if idx is not None and idx < len(docs):
                        doc = docs[idx]
                        # Ajouter le score dans les métadonnées
                        doc.metadata["rerank_score"] = float(score)
                        reranked_docs.append(doc)

            if reranked_docs:
                return reranked_docs
            else:
                if config.VERBOSE:
                    print(f"⚠️ Format rerank inconnu: {results}")
                return docs[:top_k]

        else:
            if config.VERBOSE:
                print(f"⚠️ Rerank error {response.status_code}")
            return docs[:top_k]

    except Exception as e:
        if config.VERBOSE:
            print(f"⚠️ Rerank failed: {e}")
        return docs[:top_k]


def rag_query(
    query: str, retriever, llm, logger=None, top_k: int = None, mode: str = None
):
    """
    Exécute une requête RAG complète avec le mode de prompt configuré.

    Args:
        query: Question de l'utilisateur
        retriever: Retriever ChromaDB
        llm: Modèle LLM
        logger: Logger optionnel
        top_k: Nombre de documents finaux
        mode: Mode de prompt à utiliser (override config.PROMPT_MODE)

    Returns:
        dict: Résultat avec answer, sources, scores, etc.
    """
    start_time = time.time()
    top_k = top_k or config.RAG_TOP_K_DOCS

    # Déterminer le mode à utiliser
    prompt_mode = mode or config.PROMPT_MODE

    try:
        if config.VERBOSE:
            print(f"\n{'=' * 60}")
            print(f"🔍 Nouvelle requête (mode: {prompt_mode})")
            print(f"{'=' * 60}")
            print(f"❓ Question: {query}\n")

        # 1. HyDE
        if config.VERBOSE:
            print("⏳ 1/4: Génération HyDE...")
        hyde_query = generate_hyde(query, llm)

        # 2. Retrieval
        if config.VERBOSE:
            print(f"⏳ 2/4: Retrieval (top {config.RAG_TOP_N_RETRIEVAL})...")
        docs_initial = retriever.invoke(hyde_query)

        if config.VERBOSE:
            print(f"   ✓ {len(docs_initial)} documents récupérés")

        # 3. Rerank
        if config.VERBOSE:
            print(f"⏳ 3/4: Rerank (top {top_k})...")
        docs_final = rerank_documents(query, docs_initial, top_k)

        if config.VERBOSE:
            print(f"   ✓ {len(docs_final)} documents après rerank")

        # 4. Génération de la réponse avec le prompt du mode
        if config.VERBOSE:
            print(f"⏳ 4/4: Génération de la réponse (mode: {prompt_mode})...\n")

        context = "\n\n".join([d.page_content for d in docs_final])

        # Utiliser le template RAG du mode configuré
        rag_template = config.get_prompt_template("rag", mode=prompt_mode)
        rag_prompt = PromptTemplate.from_template(rag_template)

        chain = rag_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": query})

        execution_time = time.time() - start_time

        # Préparer métadonnées
        sources = [d.metadata.get("source", "N/A") for d in docs_final]
        rerank_scores = [d.metadata.get("rerank_score", None) for d in docs_final]

        if config.VERBOSE:
            print(f"📄 Réponse générée:")
            print(f"   {answer[:300]}...")
            print(f"\n📁 Sources utilisées:")
            for i, (src, score) in enumerate(zip(sources, rerank_scores), 1):
                score_str = f"[{score:.3f}]" if score else "[N/A]"
                print(f"   {i}. {score_str} {src}")
            print(f"\n⏱️  Temps total: {execution_time:.2f}s")
            print(f"{'=' * 60}\n")

        return {
            "answer": answer,
            "sources": sources,
            "rerank_scores": rerank_scores,
            "n_docs_retrieved": len(docs_initial),
            "n_docs_final": len(docs_final),
            "execution_time": execution_time,
            "prompt_mode": prompt_mode,
            "error": None,
            "hyde_query": hyde_query,
            "retrieved_docs": docs_initial,
            "reranked_docs": docs_final,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)

        if config.VERBOSE:
            print(f"❌ Erreur: {error_msg}")

        return {
            "answer": None,
            "sources": [],
            "rerank_scores": [],
            "n_docs_retrieved": 0,
            "n_docs_final": 0,
            "execution_time": execution_time,
            "prompt_mode": prompt_mode,
            "error": error_msg,
            "hyde_query": hyde_query,
            "retrieved_docs": docs_initial,
            "reranked_docs": docs_final,
        }


def rag_query_stream(
    query: str, retriever, llm, logger=None, topk: int = None, mode: str = None
):
    """
    Excute une requte RAG avec streaming de la rponse.
    Yields: tuple (chunk, metadata) où metadata contient les infos à la fin
    """
    start_time = time.time()
    topk = topk or config.RAG_TOP_K_DOCS
    prompt_mode = mode or config.PROMPT_MODE

    full_answer = ""
    docs_initial = []
    docs_final = []
    hyde_query = query

    try:
        if config.VERBOSE:
            print(f"Query streaming, mode {prompt_mode}: {query}")

        # HyDE
        hyde_query = generate_hyde(query, llm)

        # Retrieval
        docs_initial = retriever.invoke(hyde_query)

        # Rerank
        docs_final = rerank_documents(query, docs_initial, topk)

        # Génération STREAMÉE
        context = "\n".join([d.page_content for d in docs_final])
        rag_template = config.get_prompt_template("rag", mode=prompt_mode)
        rag_prompt = PromptTemplate.from_template(rag_template)
        chain = rag_prompt | llm

        sources = [d.metadata.get("source", "N/A") for d in docs_final]
        rerank_scores = [d.metadata.get("rerank_score", None) for d in docs_final]

        # STREAM les chunks
        for chunk in chain.stream({"context": context, "query": query}):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += content
            yield {"type": "chunk", "content": content}

        execution_time = time.time() - start_time

        # Yielder les métadonnées à la fin
        yield {
            "type": "metadata",
            "sources": sources,
            "rerank_scores": rerank_scores,
            "ndocs_retrieved": len(docs_initial),
            "ndocs_final": len(docs_final),
            "execution_time": execution_time,
            "prompt_mode": prompt_mode,
            "error": None,
            "hyde_query": hyde_query,
            "retrieved_docs": docs_initial,
            "reranked_docs": docs_final,
        }

    except Exception as e:
        error_msg = f"Erreur: {e}"
        yield {"type": "error", "content": error_msg}
