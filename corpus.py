# corpus.py
"""Organisation d'un corpus riche en documents similaires.

Le corpus réunit, pour chaque réunion (dossier « 08 04 2024 »), des brouillons
et leurs versions validées, et une même note revient souvent d'une réunion à
l'autre. Ce module applique les règles usuelles pour ce cas :

- métadonnées tirées du chemin : date de réunion, statut validé, titre et
  famille de document (même note quelles que soient sa version et son format) ;
- en-tête contextuel ajouté devant chaque passage avant indexation, pour que
  l'embedding et BM25 distinguent des passages presque identiques ;
- regroupement des passages quasi identiques à la recherche, en gardant la
  version validée la plus récente ;
- plafond de passages par document, pour diversifier les sources ;
- filtres par période de réunion et par statut.
"""

import fnmatch
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_core.documents import Document

import config

# Marqueurs de version et de date retirés pour identifier la famille d'un document
_VERSION_TOKEN = re.compile(r"^(v\d+|version|vf|def|final[e]?)$")
_DATE_TOKEN = re.compile(r"^\d{6,8}$")


def fold(text: str) -> str:
    """Minuscules sans accents (comparaisons tolérantes)."""
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


# ============================================================================
# MÉTADONNÉES TIRÉES DU CHEMIN
# ============================================================================


def meeting_date(path: str) -> int:
    """
    Date de réunion tirée du dossier le plus proche qui en contient une.

    Args:
        path: Chemin du fichier

    Returns:
        int: Date au format AAAAMMJJ, 0 si aucune date reconnue
    """
    pattern = re.compile(config.MEETING_DATE_PATTERN)
    for part in reversed(Path(path).parent.parts):
        match = pattern.search(part)
        if match:
            day, month, year = (int(g) for g in match.groups())
            if 1 <= day <= 31 and 1 <= month <= 12:
                return year * 10000 + month * 100 + day
    return 0


def format_meeting_date(value: Any) -> str:
    """AAAAMMJJ -> « JJ/MM/AAAA » (chaîne vide si inconnue)."""
    try:
        value = int(value or 0)
    except (TypeError, ValueError):
        return ""
    if not value:
        return ""
    return f"{value % 100:02d}/{value // 100 % 100:02d}/{value // 10000}"


def is_validated(path: str) -> bool:
    """
    Vrai si le nom du fichier ou d'un dossier parent signale une version validée
    (mots de documents.mots_version_validee, sans accents ni majuscules).
    """
    folded = fold(str(Path(path).with_suffix("")))
    return any(fold(word) in folded for word in config.VALIDATED_KEYWORDS)


def document_title(path: str) -> str:
    """
    Titre lisible : nom du fichier sans numéro d'ordre (« 5- ») ni soulignés.

    Args:
        path: Chemin du fichier

    Returns:
        str: Titre
    """
    stem = Path(path).stem.replace("_", " ")
    stem = re.sub(r"^\s*\d{1,2}\s*[-.)]\s*", "", stem)
    return re.sub(r"\s+", " ", stem).strip()


def document_family(path: str) -> str:
    """
    Clé de famille : même valeur pour les versions et formats d'une même note
    (« 5- Note Préfet PN -20240404.pdf » et « Note Préfet PN -20240404.odt »,
    « 3- PAR V2 » et « PAR V2 »).

    Args:
        path: Chemin du fichier

    Returns:
        str: Clé normalisée (sans accents, numéros d'ordre, dates ni versions)
    """
    tokens = re.findall(r"[a-z0-9]+", fold(document_title(path)))
    kept = [
        t for t in tokens
        if not _VERSION_TOKEN.match(t) and not _DATE_TOKEN.match(t) and not (t.isdigit() and len(t) <= 2)
    ]
    return " ".join(kept) or fold(Path(path).stem)


def is_excluded(path: str) -> bool:
    """Vrai si le nom du fichier correspond à un motif de documents.exclure."""
    name = Path(path).name.lower()
    return any(fnmatch.fnmatch(name, pattern.lower()) for pattern in config.DOCUMENT_EXCLUDE_PATTERNS)


def document_metadata(path: str) -> Dict[str, Any]:
    """
    Métadonnées de corpus d'un fichier, ajoutées à chacun de ses passages.

    Args:
        path: Chemin du fichier

    Returns:
        dict: meeting_date (int AAAAMMJJ, 0 si inconnue), validated (bool),
        doc_title (str), doc_family (str)
    """
    return {
        "meeting_date": meeting_date(path),
        "validated": is_validated(path),
        "doc_title": document_title(path),
        "doc_family": document_family(path),
    }


# ============================================================================
# EN-TÊTE CONTEXTUEL
# ============================================================================


def context_header(metadata: Dict[str, Any]) -> str:
    """
    En-tête placé devant un passage avant indexation.

    Exemple : « [Note Préfet PN -20240404 | réunion du 08/04/2024 | version validée] »

    Args:
        metadata: Métadonnées du passage

    Returns:
        str: En-tête suivi d'un saut de ligne, ou chaîne vide si désactivé
    """
    if not config.CONTEXT_HEADER:
        return ""
    parts = [metadata.get("doc_title") or Path(metadata.get("source", "")).stem]
    if metadata.get("attachment"):
        parts[0] += f" > {metadata['attachment']}"
    date = format_meeting_date(metadata.get("meeting_date"))
    if date:
        parts.append(f"réunion du {date}")
    parts.append("version validée" if metadata.get("validated") else "version de travail")
    return "[" + " | ".join(parts) + "]\n"


def chunk_body(doc: Document) -> str:
    """Texte d'un passage sans son en-tête contextuel."""
    header_len = int(doc.metadata.get("header_len", 0) or 0)
    return doc.page_content[header_len:]


# ============================================================================
# DOCUMENTS SIMILAIRES À LA RECHERCHE
# ============================================================================


def shingles(text: str, n: int = 5) -> Set[int]:
    """
    Empreinte d'un texte : n-grammes de mots (hachés), insensibles à la casse,
    aux accents et à la ponctuation.
    """
    words = re.findall(r"\w+", fold(text))
    if len(words) < n:
        return {hash(w) for w in words}
    return {hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def jaccard(a: Set[int], b: Set[int]) -> float:
    """Similarité de Jaccard entre deux empreintes."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def inclusion(a: Set[int], b: Set[int]) -> float:
    """Part de la plus petite empreinte contenue dans l'autre."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def near_duplicate(a: Set[int], b: Set[int], threshold: float = None, min_size: int = None) -> bool:
    """
    Vrai si deux passages sont des versions quasi identiques d'un même texte.

    Deux critères, l'un ou l'autre suffit :
    - similarité de Jaccard >= seuil : textes presque identiques ;
    - inclusion >= seuil, si le plus petit passage compte au moins `min_size`
      empreintes : un même texte découpé différemment selon le format
      (coupures décalées entre le PDF et l'ODT d'une note). La taille minimale
      évite de traiter comme doublon un fragment banal (bloc d'adresse).

    Sur le corpus du 2026-10-09, l'inclusion détecte 67 % de doublons de plus
    entre versions d'une même note que Jaccard seul.

    Args:
        a, b: Empreintes (corpus.shingles)
        threshold: Seuil (défaut : config.DEDUP_SIMILARITY)
        min_size: Taille minimale pour l'inclusion (défaut : config.DEDUP_MIN_SHINGLES)
    """
    threshold = config.DEDUP_SIMILARITY if threshold is None else threshold
    min_size = config.DEDUP_MIN_SHINGLES if min_size is None else min_size
    if jaccard(a, b) >= threshold:
        return True
    return min(len(a), len(b)) >= min_size and inclusion(a, b) >= threshold


def _preference(doc: Document, rank: int) -> tuple:
    """Ordre de préférence dans un groupe : validée, puis récente, puis mieux classée."""
    return (bool(doc.metadata.get("validated")), int(doc.metadata.get("meeting_date") or 0), -rank)


def short_label(metadata: Dict[str, Any]) -> str:
    """Libellé court d'un document : nom de fichier et date de réunion."""
    label = Path(metadata.get("source", "")).name
    date = format_meeting_date(metadata.get("meeting_date"))
    return f"{label} ({date})" if date else label


def collapse_duplicates(docs: List[Document], threshold: float = None) -> List[Document]:
    """
    Regroupe les passages quasi identiques et garde un représentant par groupe.

    Le représentant est la version validée la plus récente (à défaut la mieux
    classée) ; les autres membres sont listés dans sa métadonnée `aussi_dans`.
    Les groupes gardent l'ordre de leur membre le mieux classé.

    Args:
        docs: Passages classés par pertinence décroissante
        threshold: Similarité minimale (défaut : config.DEDUP_SIMILARITY)

    Returns:
        List[Document]: Représentants (copies), dans l'ordre de classement
    """
    threshold = config.DEDUP_SIMILARITY if threshold is None else threshold
    groups: List[Dict[str, Any]] = []
    for rank, doc in enumerate(docs):
        prints = shingles(chunk_body(doc))
        for group in groups:
            if near_duplicate(prints, group["prints"], threshold):
                group["members"].append((rank, doc))
                break
        else:
            groups.append({"prints": prints, "members": [(rank, doc)]})

    result = []
    for group in groups:
        rank, best = max(group["members"], key=lambda m: _preference(m[1], m[0]))
        others = []
        for _, member in group["members"]:
            if member is best:
                continue
            label = short_label(member.metadata)
            if label not in others and label != short_label(best.metadata):
                others.append(label)
        metadata = dict(best.metadata)
        if others:
            metadata["aussi_dans"] = " ; ".join(others)
        result.append(Document(page_content=best.page_content, metadata=metadata))
    return result


def cap_per_document(docs: List[Document], max_per_document: int = None) -> List[Document]:
    """
    Limite le nombre de passages d'un même document.

    Un document est identifié par sa famille et sa date de réunion : un nom
    générique (« Note Préfet ») ne regroupe ainsi pas deux notes distinctes.
    Les copies d'une même note d'une réunion à l'autre sont, elles, déjà
    fusionnées par collapse_duplicates.

    Args:
        docs: Passages classés
        max_per_document: Plafond (défaut : config.MAX_PASSAGES_PER_DOCUMENT ; 0 = sans limite)

    Returns:
        List[Document]: Passages conservés, dans l'ordre
    """
    limit = config.MAX_PASSAGES_PER_DOCUMENT if max_per_document is None else max_per_document
    if limit <= 0:
        return list(docs)
    counts: Dict[str, int] = {}
    kept = []
    for doc in docs:
        family = doc.metadata.get("doc_family") or doc.metadata.get("source", "")
        key = f"{family}|{doc.metadata.get('meeting_date', 0)}"
        if counts.get(key, 0) < limit:
            counts[key] = counts.get(key, 0) + 1
            kept.append(doc)
    return kept


# ============================================================================
# FILTRES
# ============================================================================


def build_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Clause `where` Chroma correspondant aux filtres de l'interface.

    Args:
        filters: {"date_min": AAAAMMJJ, "date_max": AAAAMMJJ, "validated_only": bool}

    Returns:
        dict ou None si aucun filtre actif
    """
    if not filters:
        return None
    conditions = []
    if filters.get("date_min"):
        conditions.append({"meeting_date": {"$gte": int(filters["date_min"])}})
    if filters.get("date_max"):
        conditions.append({"meeting_date": {"$lte": int(filters["date_max"])}})
    if filters.get("validated_only"):
        conditions.append({"validated": True})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def matches_filters(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Vrai si des métadonnées satisfont les filtres (même logique que build_where)."""
    if not filters:
        return True
    date = int(metadata.get("meeting_date") or 0)
    if filters.get("date_min") and date < int(filters["date_min"]):
        return False
    if filters.get("date_max") and date > int(filters["date_max"]):
        return False
    if filters.get("validated_only") and not metadata.get("validated"):
        return False
    return True
