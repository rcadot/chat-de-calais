# retrieval.py
"""Recherche hybride : vectorielle (sémantique) et BM25 (mots-clés), fusion RRF.

La recherche vectorielle retrouve les passages proches par le sens ; BM25
retrouve les correspondances exactes (numéros d'article, Cerfa, sigles, noms
de communes) que les embeddings captent mal. Les deux classements sont
fusionnés par Reciprocal Rank Fusion avant le rerank.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config
import corpus as corpus_rules

log = logging.getLogger(__name__)

STOPWORDS_FR = set(
    """
    a au aux avec ce ces cet cette dans de des du elle en est et eux il ils je la
    le les leur leurs lui ma mais me meme mes moi mon ne nos notre nous on ou par
    pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre
    vous y d l j m n s t c qu ete etre avoir sont ont fait faire comme plus tout
    tous toute toutes si sans sous entre dont cela ceci quel quelle quels quelles
    """.split()
)


def tokenize_fr(text: str) -> List[str]:
    """
    Découpe un texte français en termes pour BM25.

    Minuscules, accents retirés, mots vides supprimés ; les nombres sont
    conservés (articles, Cerfa, années).

    Args:
        text: Texte à découper

    Returns:
        List[str]: Termes
    """
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS_FR and (len(t) > 1 or t.isdigit())]


def _doc_key(doc: Document) -> Tuple[Any, Any, str]:
    """Clé d'identité d'un chunk (source, page, début du texte)."""
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:200])


def reciprocal_rank_fusion(rankings: List[List[Document]], k: int = None) -> List[Document]:
    """
    Fusionne plusieurs classements par Reciprocal Rank Fusion.

    Score d'un document : somme sur les classements de 1 / (k + rang).

    Args:
        rankings: Listes de documents ordonnées par pertinence décroissante
        k: Constante de lissage (défaut : config.RRF_K)

    Returns:
        List[Document]: Documents uniques, du plus au moins pertinent
    """
    k = k or config.RRF_K
    scores: Dict[Tuple, float] = {}
    docs: Dict[Tuple, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


class BM25Index:
    """Index BM25 en mémoire sur un ensemble de chunks."""

    def __init__(self, docs: List[Document]):
        """
        Args:
            docs: Chunks à indexer
        """
        self.docs = docs
        tokenized = [tokenize_fr(d.page_content) for d in docs]
        self.term_sets = [set(tokens) for tokens in tokenized]
        self.bm25 = BM25Okapi(tokenized) if docs else None

    def search(
        self, query: str, k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retourne les k chunks les mieux classés parmi ceux qui contiennent au
        moins un terme de la requête (sur un petit corpus, l'IDF d'Okapi peut
        être nul : le score seul ne suffit pas à filtrer).

        Args:
            query: Requête en langage naturel
            k: Nombre maximal de résultats
            filters: Filtres de corpus (période de réunion, versions validées)

        Returns:
            List[Document]: Chunks classés
        """
        tokens = tokenize_fr(query)
        if self.bm25 is None or not tokens:
            return []
        query_terms = set(tokens)
        matching = [
            i
            for i, terms in enumerate(self.term_sets)
            if terms & query_terms and corpus_rules.matches_filters(self.docs[i].metadata, filters)
        ]
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(matching, key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[:k]]


class HybridRetriever:
    """Retriever combinant recherche vectorielle et BM25."""

    supports_keyword_query = True

    def __init__(self, vector_retriever: Any, bm25_index: BM25Index, k: int = None):
        """
        Args:
            vector_retriever: Retriever vectoriel (méthode invoke)
            bm25_index: Index BM25 des mêmes chunks
            k: Nombre de documents renvoyés après fusion
        """
        self.vector_retriever = vector_retriever
        self.bm25_index = bm25_index
        self.k = k or config.RAG_TOP_N_RETRIEVAL
        self._by_position = {
            (d.metadata.get("source"), d.metadata.get("chunk_index")): d
            for d in bm25_index.docs
            if d.metadata.get("chunk_index") is not None
        }

    def get_neighbors(
        self, doc: Document, window: int
    ) -> Tuple[List[Document], List[Document]]:
        """
        Retourne les chunks qui précèdent et suivent un chunk dans son fichier.

        Args:
            doc: Chunk de référence (métadonnées source et chunk_index)
            window: Nombre de voisins de chaque côté

        Returns:
            (avant, après): Chunks voisins dans l'ordre de lecture
        """
        source, index = doc.metadata.get("source"), doc.metadata.get("chunk_index")
        if index is None or window <= 0:
            return [], []
        before = [self._by_position.get((source, index - k)) for k in range(window, 0, -1)]
        after = [self._by_position.get((source, index + k)) for k in range(1, window + 1)]
        return [d for d in before if d], [d for d in after if d]

    def invoke(
        self,
        query: str,
        keyword_query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Recherche hybride.

        Args:
            query: Requête pour la recherche vectorielle (éventuellement
                enrichie par HyDE)
            keyword_query: Requête pour BM25, en pratique la question
                d'origine, plus précise que le document HyDE (défaut : query)
            filters: Filtres de corpus (période de réunion, versions validées)

        Returns:
            List[Document]: Documents fusionnés
        """
        vector_docs = vector_search(self.vector_retriever, query, filters)
        keyword_docs = self.bm25_index.search(
            keyword_query or query, config.RETRIEVER_TOP_K, filters=filters
        )
        return reciprocal_rank_fusion([vector_docs, keyword_docs])[: self.k]

    def meeting_dates(self) -> List[int]:
        """Dates de réunion présentes dans l'index (AAAAMMJJ, triées)."""
        return sorted({int(d.metadata.get("meeting_date") or 0) for d in self.bm25_index.docs} - {0})


def vector_search(retriever: Any, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    Recherche vectorielle, filtrée si des filtres de corpus sont actifs.

    Args:
        retriever: Retriever vectoriel LangChain (attribut vectorstore)
        query: Requête
        filters: Filtres de corpus

    Returns:
        List[Document]: Documents trouvés
    """
    where = corpus_rules.build_where(filters)
    store = getattr(retriever, "vectorstore", None)
    if where is None or store is None:
        return retriever.invoke(query)
    return store.similarity_search(query, k=config.RETRIEVER_TOP_K, filter=where)


def available_meeting_dates(retriever: Any) -> List[int]:
    """
    Dates de réunion présentes dans l'index d'un retriever (pour les filtres).

    Args:
        retriever: Retriever hybride, combiné ou vectoriel

    Returns:
        List[int]: Dates AAAAMMJJ triées (liste vide si inconnues)
    """
    for candidate in [retriever] + list(getattr(retriever, "retrievers", [])):
        if callable(getattr(candidate, "meeting_dates", None)):
            return candidate.meeting_dates()
    store = getattr(retriever, "vectorstore", None)
    if store is not None:
        metadatas = store.get(include=["metadatas"]).get("metadatas") or []
        return sorted({int((m or {}).get("meeting_date") or 0) for m in metadatas} - {0})
    return []


def make_retriever(vectorstore: Any, docs: Optional[List[Document]] = None) -> Any:
    """
    Construit le retriever d'un vectorstore : hybride si USE_HYBRID_SEARCH.

    Args:
        vectorstore: Vectorstore Chroma (langchain_chroma)
        docs: Chunks du vectorstore s'ils sont déjà en mémoire ; sinon ils
            sont relus depuis la collection

    Returns:
        Retriever vectoriel ou HybridRetriever
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})
    if not config.USE_HYBRID_SEARCH:
        return retriever

    if docs is None:
        data = vectorstore.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text or "", metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
    log.info("Index BM25 : %d chunks", len(docs))
    return HybridRetriever(retriever, BM25Index(docs))


def merge_overlapping(texts: List[str], max_overlap: int = None) -> str:
    """
    Concatène des chunks consécutifs en supprimant leur chevauchement.

    Le splitter fait se recouvrir les chunks (CHUNK_OVERLAP) : sans cette
    fusion, le recouvrement apparaîtrait deux fois dans le contexte.

    Args:
        texts: Textes de chunks consécutifs, dans l'ordre de lecture
        max_overlap: Chevauchement maximal recherché (défaut : CHUNK_OVERLAP + 50)

    Returns:
        str: Texte fusionné
    """
    max_overlap = max_overlap or config.CHUNK_OVERLAP + 50
    merged = ""
    for text in texts:
        if not merged:
            merged = text
            continue
        for k in range(min(len(merged), len(text), max_overlap), 0, -1):
            if merged.endswith(text[:k]):
                merged += text[k:]
                break
        else:
            merged += "\n" + text
    return merged
