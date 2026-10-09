# evaluate.py
"""Évaluation de la recherche documentaire sur un jeu de questions de référence.

Mesure, pour chaque question de `evaluation/questions.yaml`, si les sources
attendues sont retrouvées, et à quel rang. Permet de comparer objectivement
des réglages (HyDE, recherche hybride, rerank, seuils).

Exemples :
    python evaluate.py
    python evaluate.py --no-hyde --no-hybrid
    python evaluate.py --no-dedup          # sans regroupement des documents similaires
    python evaluate.py --generate          # produit aussi les réponses
    python evaluate.py --base _tmp/chroma_full   # autre base Chroma
"""

import argparse
import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import config
import corpus

log = logging.getLogger(__name__)


def load_questions(path: str) -> List[Dict[str, Any]]:
    """
    Charge le jeu de questions.

    Args:
        path: Chemin du fichier YAML

    Returns:
        List[dict]: Questions avec leurs sources attendues
    """
    with open(path, encoding="utf-8") as f:
        items = yaml.safe_load(f) or []
    for item in items:
        item["sources_attendues"] = [s.lower() for s in item.get("sources_attendues") or []]
    return items


def first_match_rank(docs: List[Any], expected: List[str]) -> Optional[int]:
    """
    Rang (base 1) du premier document dont la source correspond à une source attendue.

    Args:
        docs: Documents classés
        expected: Fragments de noms de fichiers attendus (minuscules)

    Returns:
        int ou None si aucun document ne correspond
    """
    for rank, doc in enumerate(docs, start=1):
        name = Path(doc.metadata.get("source", "")).name.lower()
        attachment = str(doc.metadata.get("attachment") or "").lower()
        if any(e in name or (attachment and e in attachment) for e in expected):
            return rank
    return None


# Critère de mesure des doublons, fixe quelle que soit la variante évaluée
# (sinon une variante qui assouplit le dédoublonnage assouplirait aussi sa mesure)
DUPLICATE_MEASURE_THRESHOLD = 0.85
DUPLICATE_MEASURE_MIN_SIZE = 80


def count_duplicates(docs: List[Any]) -> int:
    """
    Nombre de passages finaux quasi identiques à un passage mieux classé
    (versions successives d'un même texte occupant plusieurs places).
    """
    prints = [corpus.shingles(corpus.chunk_body(d)) for d in docs]
    return sum(
        1
        for i in range(len(prints))
        if any(
            corpus.near_duplicate(
                prints[i], prints[j], DUPLICATE_MEASURE_THRESHOLD, DUPLICATE_MEASURE_MIN_SIZE
            )
            for j in range(i)
        )
    )


def success_at_threshold(state: Dict[str, Any], expected: List[str], threshold: float) -> bool:
    """
    Succès qu'on aurait obtenu avec un autre seuil de pertinence, recalculé sur
    les passages déjà reranqués (sans nouvel appel à l'API).
    """
    relevant = [
        d for d in state["docs_reranked"]
        if d.metadata.get("rerank_score") is None or d.metadata["rerank_score"] >= threshold
    ]
    final = corpus.cap_per_document(relevant)[: config.RAG_TOP_K_DOCS]
    if expected:
        return first_match_rank(final, expected) is not None
    return not final


def evaluate(
    questions: List[Dict[str, Any]],
    retriever,
    llm,
    generate: bool,
    thresholds: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Évalue chaque question et retourne une ligne de résultats par question.

    Args:
        questions: Jeu de questions
        retriever: Retriever de la base
        llm: Modèle LLM
        generate: Produire aussi la réponse (plus lent, consomme des jetons)
        thresholds: Seuils de pertinence supplémentaires à simuler

    Returns:
        List[dict]: Résultats détaillés
    """
    from rag_pipeline import _answer_prompt, retrieve_passages
    from langchain_core.output_parsers import StrOutputParser

    rows = []
    for i, item in enumerate(questions, start=1):
        question, expected = item["question"], item["sources_attendues"]
        log.warning("[%d/%d] %s", i, len(questions), question)
        start = time.time()
        state = retrieve_passages(question, retriever, llm)
        duration = time.time() - start

        rank_retrieved = first_match_rank(state["docs_initial"], expected) if expected else None
        rank_final = first_match_rank(state["docs_final"], expected) if expected else None
        if expected:
            success = rank_final is not None
        else:  # question hors sujet : succès si rien n'est retenu
            success = not state["docs_final"]

        scores = [d.metadata.get("rerank_score") for d in state["docs_final"]]
        scores = [s for s in scores if s is not None]
        row = {
            "question": question,
            "sources_attendues": " ; ".join(expected) or "(aucune)",
            "succes": success,
            "rang_recherche": rank_retrieved,
            "rang_final": rank_final,
            "n_passages": len(state["docs_final"]),
            "meilleur_score": round(max(scores), 4) if scores else None,
            "documents_distincts": len(
                {(d.metadata.get("doc_family") or d.metadata.get("source"), d.metadata.get("meeting_date"))
                 for d in state["docs_final"]}
            ),
            "doublons": count_duplicates(state["docs_final"]),
            "duree_s": round(duration, 1),
            "sources_retenues": " ; ".join(
                Path(d.metadata.get("source", "")).name for d in state["docs_final"]
            ),
        }
        for threshold in thresholds or []:
            row[f"succes_seuil_{threshold}"] = success_at_threshold(state, expected, threshold)
        if generate and state["docs_final"]:
            chain = _answer_prompt(config.PROMPT_MODE) | llm | StrOutputParser()
            row["reponse"] = chain.invoke(
                {"context": state["context"], "query": state["standalone_query"]}
            )
        rows.append(row)
    return rows


def summarize(rows: List[Dict]) -> Dict[str, float]:
    """
    Indicateurs globaux.

    Returns:
        dict: taux de succès, rappel de la recherche (avant rerank), MRR final
        (moyenne de 1/rang de la première bonne source, 0 si absente)
    """
    with_sources = [r for r in rows if r["sources_attendues"] != "(aucune)"]
    off_topic = [r for r in rows if r["sources_attendues"] == "(aucune)"]
    n = max(len(with_sources), 1)
    return {
        "questions": len(rows),
        "succes_global": sum(r["succes"] for r in rows) / max(len(rows), 1),
        "rappel_recherche": sum(r["rang_recherche"] is not None for r in with_sources) / n,
        "succes_final": sum(r["succes"] for r in with_sources) / n,
        "mrr_final": sum(1 / r["rang_final"] for r in with_sources if r["rang_final"]) / n,
        "hors_sujet_rejetes": (
            sum(r["succes"] for r in off_topic) / len(off_topic) if off_topic else float("nan")
        ),
        "passages_moyens": sum(r["n_passages"] for r in rows) / max(len(rows), 1),
        "documents_distincts_moyens": (
            sum(r["documents_distincts"] for r in with_sources) / n
        ),
        "doublons_moyens": sum(r["doublons"] for r in rows) / max(len(rows), 1),
        "duree_moyenne_s": sum(r["duree_s"] for r in rows) / max(len(rows), 1),
        **{
            f"{key} (questions documentées / hors sujet)": (
                f"{sum(r[key] for r in with_sources) / n:.2f} / "
                f"{sum(r[key] for r in off_topic) / max(len(off_topic), 1):.2f}"
            )
            for key in (rows[0] if rows else {})
            if key.startswith("succes_seuil_")
        },
    }


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", default="evaluation/questions.yaml")
    parser.add_argument("--no-hyde", action="store_true", help="Désactive HyDE")
    parser.add_argument("--no-hybrid", action="store_true", help="Recherche vectorielle seule")
    parser.add_argument("--no-rerank", action="store_true", help="Désactive le rerank")
    parser.add_argument("--no-dedup", action="store_true", help="Sans regroupement des documents similaires")
    parser.add_argument("--max-per-doc", type=int, help="Plafond de passages par document à tester (0 = sans)")
    parser.add_argument(
        "--doublons-taille-min", type=int,
        help="Taille minimale pour le critère d'inclusion des doublons (très grand = Jaccard seul)",
    )
    parser.add_argument("--base", help="Dossier de la base Chroma à évaluer (défaut : base.dossier)")
    parser.add_argument("--label", default="", help="Libellé ajouté au nom du fichier de résultats")
    parser.add_argument(
        "--seuils", default="",
        help="Seuils de pertinence à simuler sans nouvel appel, séparés par des virgules (ex. 0.005,0.03)",
    )
    parser.add_argument("--min-score", type=float, help="Seuil RERANK_MIN_SCORE à tester")
    parser.add_argument("--generate", action="store_true", help="Produit aussi les réponses")
    args = parser.parse_args()

    config.VERBOSE = False
    config.setup_logging()
    config.USE_HYDE = config.USE_HYDE and not args.no_hyde
    config.USE_HYBRID_SEARCH = config.USE_HYBRID_SEARCH and not args.no_hybrid
    config.USE_RERANK = config.USE_RERANK and not args.no_rerank
    if args.min_score is not None:
        config.RERANK_MIN_SCORE = args.min_score
    config.DEDUP_ENABLED = config.DEDUP_ENABLED and not args.no_dedup
    if args.max_per_doc is not None:
        config.MAX_PASSAGES_PER_DOCUMENT = args.max_per_doc
    if args.doublons_taille_min is not None:
        config.DEDUP_MIN_SHINGLES = args.doublons_taille_min
    if args.base:
        config.CHROMA_DB_PATH = args.base
        config.OCR_CACHE_DIR = os.path.join(args.base, "ocr_cache")

    from albert_client import get_embeddings, get_llm
    from indexer import index_documents

    _, retriever = index_documents(get_embeddings())
    thresholds = [float(t) for t in args.seuils.split(",") if t.strip()]
    rows = evaluate(load_questions(args.questions), retriever, get_llm(), args.generate, thresholds)

    os.makedirs("evaluation/resultats", exist_ok=True)
    reglages = (
        f"hyde{int(config.USE_HYDE)}_hybride{int(config.USE_HYBRID_SEARCH)}"
        f"_rerank{int(config.USE_RERANK)}_dedup{int(config.DEDUP_ENABLED)}"
        f"_plafond{config.MAX_PASSAGES_PER_DOCUMENT}_seuil{config.RERANK_MIN_SCORE}"
        + (f"_{args.label}" if args.label else "")
    )
    out = f"evaluation/resultats/{datetime.now():%Y%m%d_%H%M}_{reglages}.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row)) or ["question"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRéglages : {reglages}")
    for key, value in summarize(rows).items():
        print(f"  {key:22s} {value:.2f}" if isinstance(value, float) else f"  {key:22s} {value}")
    print(f"\nDétail : {out}")


if __name__ == "__main__":
    main()
