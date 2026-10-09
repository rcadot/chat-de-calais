# rag_pipeline.py
"""Pipeline RAG : reformulation, HyDE, recherche hybride, rerank, génération.

Étapes communes (`iter_retrieval_steps`, ou `retrieve_passages` sans événements) :
1. reformulation de la question à partir de l'historique (questions de suivi) ;
2. HyDE : document hypothétique pour la recherche vectorielle ;
3. recherche (vectorielle, ou hybride vectorielle + BM25), filtrable par
   période de réunion et statut ;
4. regroupement des passages quasi identiques (documents similaires) ;
5. rerank Albert, seuil de pertinence, plafond de passages par document ;
6. contexte élargi aux chunks voisins, numéroté pour les citations [n].

`rag_query_stream` génère la réponse en flux ; `rag_query` en est la
version non streamée (même code, réponse assemblée).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import config
import corpus
from loaders import normalize_ocr_text
from retrieval import merge_overlapping, vector_search

log = logging.getLogger(__name__)


# ============================================================================
# CONTEXTE ET SOURCES
# ============================================================================


def source_label(metadata: Dict[str, Any]) -> str:
    """
    Libellé lisible d'une source : nom du fichier, pièce jointe, page.

    Args:
        metadata: Métadonnées d'un chunk

    Returns:
        str: Par exemple « note.pdf, p. 3 » ou « mail.eml > annexe.pdf, p. 1 »
    """
    name = Path(metadata.get("source", "inconnu")).name
    if metadata.get("attachment"):
        name = f"{name} > {metadata['attachment']}"
    page = metadata.get("page")
    return f"{name}, p. {int(page) + 1}" if page is not None else name


def source_details(metadata: Dict[str, Any]) -> str:
    """
    Date de réunion et statut d'une source, pour le contexte et l'affichage.

    Returns:
        str: Par exemple « réunion du 08/04/2024 | version validée »
    """
    parts = []
    date = corpus.format_meeting_date(metadata.get("meeting_date"))
    if date:
        parts.append(f"réunion du {date}")
    if "validated" in metadata:
        parts.append("version validée" if metadata.get("validated") else "version de travail")
    return " | ".join(parts)


def expand_with_neighbors(docs: List[Document], retriever: Any) -> List[str]:
    """
    Élargit chaque passage à ses chunks voisins (CONTEXT_NEIGHBORS de chaque côté).

    Sans index des voisins (retriever non hybride, chunks indexés avant la
    numérotation), le passage est conservé tel quel. Les textes renvoyés
    sont sans en-tête contextuel (corpus.chunk_body).

    Args:
        docs: Passages retenus
        retriever: Retriever exposant éventuellement get_neighbors

    Returns:
        List[str]: Texte élargi de chaque passage
    """
    get_neighbors = getattr(retriever, "get_neighbors", None)
    if config.CONTEXT_NEIGHBORS <= 0 or not callable(get_neighbors):
        return [corpus.chunk_body(d) for d in docs]

    texts = []
    for doc in docs:
        try:
            before, after = get_neighbors(doc, config.CONTEXT_NEIGHBORS)
        except Exception as e:
            log.warning("Voisins indisponibles : %s", e)
            before, after = [], []
        sequence = [corpus.chunk_body(d) for d in before + [doc] + after]
        texts.append(merge_overlapping(sequence))
    return texts


def format_context(docs: List[Document], texts: Optional[List[str]] = None) -> str:
    """
    Construit le contexte transmis au LLM : extraits numérotés avec leur source.

    Args:
        docs: Passages retenus après rerank
        texts: Textes à utiliser (contexte élargi) ; défaut : contenu des passages

    Returns:
        str: Contexte avec en-têtes « [n] nom_du_document, p. N | réunion du
        JJ/MM/AAAA | version validée »
    """
    texts = texts if texts is not None else [corpus.chunk_body(d) for d in docs]
    blocks = []
    for n, (doc, text) in enumerate(zip(docs, texts), start=1):
        header = " | ".join(filter(None, [source_label(doc.metadata), source_details(doc.metadata)]))
        blocks.append(f"[{n}] {header}\n{normalize_ocr_text(text)}")
    return "\n\n---\n\n".join(blocks)


def build_passages(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Décrit les passages cités pour l'interface (numéros identiques au contexte).

    Args:
        docs: Passages retenus après rerank

    Returns:
        List[dict]: num, source, label, details (date et statut), page
        (base 0), score, excerpt, ocr, ocr_images, attachment, aussi_dans
    """
    return [
        {
            "num": n,
            "source": doc.metadata.get("source", ""),
            "label": source_label(doc.metadata),
            "details": source_details(doc.metadata),
            "page": doc.metadata.get("page"),
            "score": doc.metadata.get("rerank_score"),
            "excerpt": normalize_ocr_text(corpus.chunk_body(doc)),
            "ocr": bool(doc.metadata.get("ocr")),
            "ocr_images": int(doc.metadata.get("ocr_images", 0) or 0),
            "attachment": doc.metadata.get("attachment"),
            "aussi_dans": doc.metadata.get("aussi_dans"),
        }
        for n, doc in enumerate(docs, start=1)
    ]


# ============================================================================
# ÉTAPES DU PIPELINE
# ============================================================================


def condense_question(query: str, history: Optional[List[Dict]], llm) -> str:
    """
    Reformule une question de suivi en question autonome.

    « Et pour Calais ? » devient, par exemple, « Quelle est la consommation
    d'espaces à Calais entre 2011 et 2021 ? ». Sans historique, la question
    est renvoyée telle quelle.

    Args:
        query: Nouvelle question
        history: Messages précédents [{"role": "user"|"assistant", "content": str}]
        llm: Modèle LLM

    Returns:
        str: Question autonome
    """
    if not history or config.HISTORY_TURNS <= 0:
        return query

    recent = [m for m in history if m.get("role") in ("user", "assistant")]
    recent = recent[-2 * config.HISTORY_TURNS :]
    if not recent:
        return query

    lines = []
    for message in recent:
        role = "Agent" if message["role"] == "user" else "Assistant"
        content = str(message.get("content", ""))[:600]
        lines.append(f"{role} : {content}")

    try:
        prompt = PromptTemplate.from_template(config.CONDENSE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        standalone = chain.invoke({"history": "\n".join(lines), "query": query})
        standalone = standalone.strip().strip("\"«»“” ").strip()
        if standalone:
            log.info("   Question autonome : %s", standalone)
            return standalone
    except Exception as e:
        log.warning("Reformulation impossible : %s", e)
    return query


def retrieve(
    retriever, hyde_query: str, query: str, filters: Optional[Dict[str, Any]] = None
) -> List:
    """
    Interroge le retriever : la requête HyDE sert à la recherche vectorielle,
    la question d'origine à la recherche par mots-clés (si hybride).

    Args:
        retriever: Retriever (vectoriel, hybride ou combiné)
        hyde_query: Requête enrichie par HyDE
        query: Question d'origine
        filters: Filtres de corpus (période de réunion, versions validées)

    Returns:
        List: Documents récupérés
    """
    if getattr(retriever, "supports_keyword_query", False) is True:
        if filters:
            return retriever.invoke(hyde_query, keyword_query=query, filters=filters)
        return retriever.invoke(hyde_query, keyword_query=query)
    if filters:
        return vector_search(retriever, hyde_query, filters)
    return retriever.invoke(hyde_query)


def generate_hyde(query: str, llm, mode: str = None) -> str:
    """
    Génère un document hypothétique avec HyDE en utilisant le prompt du mode configuré.

    Args:
        query: Question de l'utilisateur
        llm: Modèle LLM
        mode: Mode de prompt (None pour config.PROMPT_MODE)

    Returns:
        Query enrichie avec document hypothétique
    """
    if not config.USE_HYDE:
        return query

    try:
        hyde_template = config.get_prompt_template("hyde", mode=mode)
        hyde_prompt = PromptTemplate.from_template(hyde_template)
        chain = hyde_prompt | llm | StrOutputParser()
        hyde_doc = chain.invoke({"query": query})

        log.info("   HyDE (mode: %s): %s...", mode or config.PROMPT_MODE, hyde_doc[:200])
        return f"{query}\n\n{hyde_doc}"

    except Exception as e:
        log.warning("Erreur HyDE: %s", e)
        return query


def rerank_documents(query: str, docs: List, top_k: int = None) -> List:
    """
    Rerank les documents avec l'API Albert.

    Les documents renvoyés sont des copies portant `rerank_score` dans leurs
    métadonnées : les chunks d'origine (partagés par l'index BM25) ne sont
    pas modifiés.

    Args:
        query: Question de l'utilisateur
        docs: Liste de documents à reranker
        top_k: Nombre de documents à retourner

    Returns:
        Liste de documents triés par score décroissant, ou les top_k premiers
        documents d'origine si le rerank est désactivé ou échoue
    """
    if not config.USE_RERANK or not docs:
        return docs[:top_k] if top_k else docs

    top_k = top_k or config.RAG_TOP_K_DOCS

    try:
        doc_texts = [doc.page_content[: config.RERANK_MAX_CHARS] for doc in docs]
        log.info(" Rerank: %d docs...", len(doc_texts))

        response = requests.post(
            f"{config.ALBERT_BASE_URL}/rerank",
            json={"model": config.RERANK_MODEL, "query": query, "documents": doc_texts},
            headers={
                "Authorization": f"Bearer {config.ALBERT_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if response.status_code != 200:
            log.warning("Rerank error %s", response.status_code)
            return docs[:top_k]

        results = response.json()
        if isinstance(results, dict):
            results = results.get("results", results.get("data", []))

        scored = []
        for r in results if isinstance(results, list) else []:
            idx = r.get("index")
            score = r.get("score", r.get("relevance_score", r.get("rerank_score", 0.0)))
            if idx is not None and idx < len(docs):
                scored.append((float(score or 0.0), idx))

        if not scored:
            log.warning("Format rerank inconnu: %s", results)
            return docs[:top_k]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Document(
                page_content=docs[idx].page_content,
                metadata={**docs[idx].metadata, "rerank_score": score},
            )
            for score, idx in scored[:top_k]
        ]

    except Exception as e:
        log.warning("Rerank failed: %s", e)
        return docs[:top_k]


def filter_relevant(docs: List[Document]) -> List[Document]:
    """
    Écarte les passages dont le score de rerank est inférieur à RERANK_MIN_SCORE.

    Les passages sans score (rerank désactivé ou en échec) sont conservés.

    Args:
        docs: Passages reranqués

    Returns:
        List[Document]: Passages suffisamment pertinents
    """
    return [
        d
        for d in docs
        if d.metadata.get("rerank_score") is None
        or d.metadata["rerank_score"] >= config.RERANK_MIN_SCORE
    ]


def _status(step: str, start: float, detail: Optional[str] = None) -> Dict[str, Any]:
    """Événement d'étape : libellé (paramètres), détail éventuel, temps écoulé."""
    event = {
        "type": "status",
        "step": step,
        "content": config.STATUS_MESSAGES.get(step, step),
        "elapsed": time.time() - start,
    }
    if detail:
        event["detail"] = detail
    return event


def iter_retrieval_steps(
    query: str,
    retriever,
    llm,
    mode: str = None,
    history: Optional[List[Dict]] = None,
    top_k: int = None,
    filters: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    start: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Étapes avant génération, annoncées au fil de l'eau.

    1. reformulation de la question de suivi (si historique) ;
    2. HyDE et recherche (hybride, filtrée) ;
    3. regroupement des passages quasi identiques (documents similaires) ;
    4. rerank, seuil de pertinence, plafond par document ;
    5. contexte élargi aux voisins, numéroté pour les citations.

    Args:
        query: Question de l'utilisateur
        retriever: Retriever
        llm: Modèle LLM
        mode: Mode de prompt
        history: Messages précédents de la conversation
        top_k: Nombre maximal de passages retenus
        filters: Filtres de corpus (période de réunion, versions validées)
        state: Dictionnaire complété au fil des étapes (résultats partiels
            disponibles en cas d'erreur)
        start: Instant de départ (pour les temps affichés)

    Yields:
        dict: Événements {"type": "status", "step", "content", "detail", "elapsed"}
    """
    state = state if state is not None else {}
    start = start or time.time()
    top_k = top_k or config.RAG_TOP_K_DOCS

    # 1. Question de suivi
    if history and config.HISTORY_TURNS > 0:
        yield _status("reformulation", start)
    state["standalone_query"] = condense_question(query, history, llm)
    standalone = state["standalone_query"]
    if standalone != query:
        yield _status("reformulation", start, f"Question comprise comme : « {standalone} »")

    # 2. Recherche
    yield _status("recherche", start)
    state["hyde_query"] = generate_hyde(standalone, llm, mode=mode)
    state["docs_initial"] = retrieve(retriever, state["hyde_query"], standalone, filters)
    n_sources = len({d.metadata.get("source") for d in state["docs_initial"]})
    log.info("%d documents récupérés", len(state["docs_initial"]))
    yield _status(
        "recherche", start,
        f"{len(state['docs_initial'])} passages candidats dans {n_sources} document(s)",
    )

    # 3. Documents similaires
    candidates = state["docs_initial"]
    if config.DEDUP_ENABLED and candidates:
        candidates = corpus.collapse_duplicates(candidates)
        merged = len(state["docs_initial"]) - len(candidates)
        if merged:
            yield _status(
                "recherche", start,
                f"{merged} passage(s) quasi identique(s) regroupé(s) (versions successives d'un même document)",
            )
    state["docs_deduplicated"] = candidates

    # 4. Tri par pertinence
    yield _status("tri", start)
    state["docs_reranked"] = rerank_documents(standalone, candidates, top_k=len(candidates) or top_k)
    relevant = filter_relevant(state["docs_reranked"])
    state["docs_final"] = corpus.cap_per_document(relevant)[:top_k]
    scores = [d.metadata.get("rerank_score") for d in state["docs_final"]]
    scores = [s for s in scores if s is not None]
    if state["docs_final"]:
        best = f", meilleur score {max(scores):.2f}" if scores else ""
        detail = f"{len(state['docs_final'])} passage(s) retenu(s){best}"
    else:
        detail = "Aucun passage suffisamment pertinent"
    log.info("%s", detail)
    yield _status("tri", start, detail)

    # 5. Contexte
    texts = expand_with_neighbors(state["docs_final"], retriever)
    state["context"] = format_context(state["docs_final"], texts)


def retrieve_passages(
    query: str,
    retriever,
    llm,
    mode: str = None,
    history: Optional[List[Dict]] = None,
    top_k: int = None,
    filters: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Étapes communes avant génération (version sans événements de iter_retrieval_steps).

    Returns:
        dict: standalone_query, hyde_query, docs_initial, docs_deduplicated,
        docs_reranked, docs_final, context
    """
    state = state if state is not None else {}
    for _ in iter_retrieval_steps(
        query, retriever, llm, mode=mode, history=history, top_k=top_k,
        filters=filters, state=state,
    ):
        pass
    return state


def _answer_prompt(mode: str) -> ChatPromptTemplate:
    """Prompt de génération : consigne système du mode, puis gabarit RAG."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", config.get_prompt_template("system", mode=mode)),
            ("human", config.get_prompt_template("rag", mode=mode)),
        ]
    )


# ============================================================================
# POINTS D'ENTRÉE
# ============================================================================


def rag_query_stream(
    query: str,
    retriever,
    llm,
    logger=None,
    topk: int = None,
    mode: str = None,
    history: Optional[List[Dict]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Exécute une requête RAG avec génération en flux.

    Args:
        query: Question de l'utilisateur
        retriever: Retriever
        llm: Modèle LLM
        logger: Inutilisé (journalisation faite par l'appelant)
        topk: Nombre de passages finaux
        mode: Mode de prompt
        history: Messages précédents (questions de suivi)
        filters: Filtres de corpus (période de réunion, versions validées)

    Yields:
        dict: événements {"type": "status", "step", "content", "detail", "elapsed"},
        {"type": "chunk", "content"}, {"type": "error", "content"}, puis toujours
        un {"type": "metadata", ...} final
    """
    start_time = time.time()
    prompt_mode = mode or config.PROMPT_MODE
    state: Dict[str, Any] = {
        "standalone_query": query,
        "hyde_query": query,
        "docs_initial": [],
        "docs_final": [],
    }
    no_relevant_docs = False
    error = None
    retrieval_time = None

    try:
        log.info("Requête (mode %s) : %s", prompt_mode, query)
        yield from iter_retrieval_steps(
            query, retriever, llm, mode=prompt_mode, history=history, top_k=topk,
            filters=filters, state=state, start=start_time,
        )
        retrieval_time = time.time() - start_time

        if not state["docs_final"]:
            # Rien de pertinent : pas de génération, réponse explicite
            no_relevant_docs = True
            yield {"type": "chunk", "content": config.NO_ANSWER_MESSAGE}
        else:
            yield _status("redaction", start_time)
            chain = _answer_prompt(prompt_mode) | llm
            for chunk in chain.stream(
                {"context": state["context"], "query": state["standalone_query"]}
            ):
                content = getattr(chunk, "content", chunk)
                content = content if isinstance(content, str) else str(content)
                yield {"type": "chunk", "content": content}

    except Exception as e:
        error = str(e)
        log.error("Erreur pipeline : %s", error)
        yield {"type": "error", "content": f"Erreur: {e}"}

    docs_final = state["docs_final"] if error is None else []
    yield {
        "type": "metadata",
        "sources": [d.metadata.get("source", "N/A") for d in docs_final],
        "rerank_scores": [d.metadata.get("rerank_score") for d in docs_final],
        "passages": build_passages(docs_final),
        "no_relevant_docs": no_relevant_docs,
        "n_docs_retrieved": len(state["docs_initial"]),
        "n_docs_final": len(state["docs_final"]),
        "retrieval_time": retrieval_time,
        "execution_time": time.time() - start_time,
        "prompt_mode": prompt_mode,
        "error": error,
        "standalone_query": state["standalone_query"],
        "hyde_query": state["hyde_query"],
        "retrieved_docs": state["docs_initial"],
        "reranked_docs": state["docs_final"],
    }


def rag_query(
    query: str,
    retriever,
    llm,
    logger=None,
    top_k: int = None,
    mode: str = None,
    history: Optional[List[Dict]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Exécute une requête RAG complète (version non streamée de rag_query_stream).

    Args:
        query: Question de l'utilisateur
        retriever: Retriever
        llm: Modèle LLM
        logger: Inutilisé (journalisation faite par l'appelant)
        top_k: Nombre de documents finaux
        mode: Mode de prompt (override config.PROMPT_MODE)
        history: Messages précédents (questions de suivi)
        filters: Filtres de corpus (période de réunion, versions validées)

    Returns:
        dict: Métadonnées de rag_query_stream et `answer` (None en cas d'erreur)
    """
    parts: List[str] = []
    metadata: Dict[str, Any] = {}
    for item in rag_query_stream(
        query, retriever, llm, topk=top_k, mode=mode, history=history, filters=filters
    ):
        if item["type"] == "chunk":
            parts.append(item["content"])
        elif item["type"] == "metadata":
            metadata = {k: v for k, v in item.items() if k != "type"}

    metadata["answer"] = None if metadata.get("error") else "".join(parts)
    return metadata
