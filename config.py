# config.py
"""Configuration du système RAG, lue depuis `parametres.yaml`.

Les valeurs modifiables (modèles, adresses, réglages de recherche, textes de
l'interface, prompts) sont dans `parametres.yaml`, commenté pour un usage sans
connaissance du code. Ce module le lit, le valide (messages d'erreur en
français indiquant la clé fautive) et expose les constantes utilisées par le
reste du code.

Ordre de priorité : variables d'environnement (`.env`), puis `parametres.yaml`.
Le fichier lu peut être changé avec la variable PARAMETRES_FILE.
"""

import logging
import os
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from dotenv import load_dotenv

load_dotenv()

PARAMETRES_FILE = os.getenv(
    "PARAMETRES_FILE", str(Path(__file__).resolve().parent / "parametres.yaml")
)


try:  # classe conservée si le module est rechargé (importlib.reload)
    ParametresError  # type: ignore[used-before-def]
except NameError:

    class ParametresError(ValueError):
        """Erreur de contenu dans le fichier de paramètres."""


# ============================================================================
# LECTURE ET VALIDATION
# ============================================================================


def load_parameters(path: str = None) -> Dict[str, Any]:
    """
    Lit le fichier de paramètres YAML.

    Args:
        path: Chemin du fichier (défaut : PARAMETRES_FILE)

    Returns:
        dict: Paramètres

    Raises:
        ParametresError: Fichier absent ou YAML invalide (avec numéro de ligne)
    """
    path = path or PARAMETRES_FILE
    if not os.path.exists(path):
        raise ParametresError(f"Fichier de paramètres introuvable : {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (ligne {mark.line + 1}, colonne {mark.column + 1})" if mark else ""
        raise ParametresError(f"{path} : syntaxe YAML invalide{where} : {e}") from e
    if not isinstance(data, dict):
        raise ParametresError(f"{path} : le fichier doit contenir des sections (albert:, recherche:...)")
    return data


def _get(params: Dict[str, Any], key: str, expected: type) -> Any:
    """
    Lit une clé pointée (« recherche.score_minimum ») et vérifie son type.

    Args:
        params: Paramètres lus
        key: Chemin de la clé, sections séparées par des points
        expected: Type attendu (int, float, bool, str, list, dict)

    Returns:
        Valeur de la clé

    Raises:
        ParametresError: Clé absente ou de mauvais type
    """
    value: Any = params
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ParametresError(
                f"{os.path.basename(PARAMETRES_FILE)} : clé « {key} » manquante"
            )
        value = value[part]

    labels = {int: "un nombre entier", float: "un nombre", bool: "true ou false",
              str: "un texte", list: "une liste [..]", dict: "une section"}
    ok = isinstance(value, expected) and not (expected in (int, float) and isinstance(value, bool))
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        ok, value = True, float(value)
    if not ok:
        raise ParametresError(
            f"{os.path.basename(PARAMETRES_FILE)} : la clé « {key} » doit être "
            f"{labels.get(expected, expected.__name__)} (valeur lue : {value!r})"
        )
    return value


def check_template(name: str, template: str, required: Iterable[str]) -> None:
    """
    Vérifie qu'un prompt contient ses variables obligatoires, et rien d'autre.

    Args:
        name: Nom du prompt (pour le message d'erreur)
        template: Texte du prompt
        required: Variables attendues (ex. {"context", "query"})

    Raises:
        ParametresError: Variable manquante, inconnue, ou accolade mal formée
    """
    required = set(required)
    try:
        fields = {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    except ValueError as e:
        raise ParametresError(
            f"Prompt « {name} » : accolade mal formée ({e}). "
            "Pour écrire une accolade, la doubler : {{ ou }}."
        ) from e
    missing = required - fields
    unknown = fields - required
    if missing:
        raise ParametresError(
            f"Prompt « {name} » : variable(s) obligatoire(s) absente(s) : "
            + ", ".join("{" + m + "}" for m in sorted(missing))
        )
    if unknown:
        raise ParametresError(
            f"Prompt « {name} » : variable(s) inconnue(s) : "
            + ", ".join("{" + u + "}" for u in sorted(unknown))
            + ". Pour écrire une accolade, la doubler : {{ ou }}."
        )


def _env_bool(name: str, default: bool) -> bool:
    """Booléen lu dans l'environnement s'il est défini, sinon valeur par défaut."""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "oui", "yes")


_P = load_parameters()

# ============================================================================
# API
# ============================================================================
ALBERT_API_KEY = os.getenv("ALBERT_API_KEY", "")
ALBERT_BASE_URL = os.getenv("ALBERT_BASE_URL", _get(_P, "albert.url", str)).rstrip("/")

# ============================================================================
# MODÈLES
# ============================================================================
EMBEDDINGS_MODEL = _get(_P, "albert.modele_embeddings", str)
LLM_MODEL = os.getenv("LLM_MODEL", _get(_P, "albert.modele_reponse", str))
RERANK_MODEL = _get(_P, "albert.modele_rerank", str)

EMBEDDINGS_CONFIG = {
    "encoding_format": "float",
    "chunk_size": _get(_P, "base.taille_lot", int),
    "max_retries": _get(_P, "albert.tentatives", int),
    "request_timeout": _get(_P, "albert.delai_max_secondes", int),
}
LLM_TEMPERATURE = _get(_P, "albert.temperature", float)

# ============================================================================
# DÉCOUPAGE
# ============================================================================
CHUNK_SIZE = _get(_P, "decoupage.taille", int)
CHUNK_OVERLAP = _get(_P, "decoupage.chevauchement", int)
CHUNK_SEPARATORS = _get(_P, "decoupage.separateurs", list)
CONTEXT_HEADER = _get(_P, "decoupage.entete_contextuel", bool)
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ParametresError("decoupage.chevauchement doit être compris entre 0 et decoupage.taille")

# ============================================================================
# DOCUMENTS ET INDEXATION
# ============================================================================
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", _get(_P, "documents.dossier", str))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", _get(_P, "base.dossier", str))
COLLECTION_NAME = _get(_P, "base.collection", str)
BATCH_SIZE = _get(_P, "base.taille_lot", int)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
SUPPORTED_EXTENSIONS = [e.lower() for e in _get(_P, "documents.extensions", list)]
DOCUMENT_EXCLUDE_PATTERNS = _get(_P, "documents.exclure", list)
VALIDATED_KEYWORDS = _get(_P, "documents.mots_version_validee", list)
MEETING_DATE_PATTERN = _get(_P, "documents.motif_date_dossier", str)
try:
    if re.compile(MEETING_DATE_PATTERN).groups != 3:
        raise ParametresError(
            "documents.motif_date_dossier doit contenir 3 groupes entre parenthèses : jour, mois, année"
        )
except re.error as e:
    raise ParametresError(f"documents.motif_date_dossier : expression invalide ({e})") from e
TEXT_ENCODINGS = _get(_P, "documents.encodages_texte", list)

# ============================================================================
# OCR (PDF scannés et images)
# ============================================================================
OCR_ENABLED = _env_bool("OCR_ENABLED", _get(_P, "ocr.active", bool))
# openweight : /chat/completions page par page (accès ouvert)
# mistral : /ocr sur le PDF entier (accès restreint, repli sur openweight si refus)
OCR_ENGINE = os.getenv("OCR_ENGINE", _get(_P, "ocr.moteur", str))
OCR_MODELS = {
    "openweight": _get(_P, "albert.modele_ocr", str),
    "mistral": _get(_P, "albert.modele_ocr_mistral", str),
}
if OCR_ENGINE not in OCR_MODELS:
    raise ParametresError(f"ocr.moteur doit valoir openweight ou mistral (valeur lue : {OCR_ENGINE!r})")
OCR_MIN_CHARS = _get(_P, "ocr.caracteres_min_page", int)
OCR_DPI = _get(_P, "ocr.dpi", int)
OCR_MAX_SIDE = _get(_P, "ocr.cote_max_pixels", int)
OCR_MAX_PAGES = _get(_P, "ocr.pages_max", int)
OCR_IMAGES_IN_TEXT_PAGES = _env_bool(
    "OCR_IMAGES_IN_TEXT_PAGES", _get(_P, "ocr.images_dans_pages", bool)
)
OCR_IMAGE_MIN_RATIO = _get(_P, "ocr.part_image_min", float)
OCR_IMAGE_MAX_RATIO = _get(_P, "ocr.part_image_max", float)
OCR_MAX_IMAGES = _get(_P, "ocr.images_max", int)
OCR_IMAGE_MIN_CHARS = _get(_P, "ocr.caracteres_min_image", int)
OCR_TIMEOUT = _get(_P, "ocr.delai_max_secondes", int)
OCR_CACHE_DIR = os.path.join(CHROMA_DB_PATH, "ocr_cache")
OCR_PROMPT = _get(_P, "prompts.ocr", str).strip()

# ============================================================================
# RECHERCHE ET PIPELINE RAG
# ============================================================================
RETRIEVER_TOP_K = _get(_P, "recherche.candidats_par_methode", int)
RAG_TOP_N_RETRIEVAL = _get(_P, "recherche.candidats_a_trier", int)
RAG_TOP_K_DOCS = _get(_P, "recherche.passages_retenus", int)
USE_HYDE = _env_bool("USE_HYDE", _get(_P, "recherche.hyde", bool))
USE_RERANK = _env_bool("USE_RERANK", _get(_P, "recherche.rerank", bool))
RERANK_MAX_CHARS = CHUNK_SIZE + 200  # texte transmis au reranker (passage et en-tête)
USE_HYBRID_SEARCH = _env_bool("USE_HYBRID_SEARCH", _get(_P, "recherche.hybride", bool))
RRF_K = _get(_P, "recherche.rrf_k", int)
RERANK_MIN_SCORE = _get(_P, "recherche.score_minimum", float)
RELEVANCE_WARNING_SCORE = _get(_P, "recherche.score_alerte", float)
CONTEXT_NEIGHBORS = _get(_P, "recherche.voisins", int)
HISTORY_TURNS = _get(_P, "recherche.tours_historique", int)
DEDUP_ENABLED = _get(_P, "recherche.dedoublonnage", bool)
DEDUP_SIMILARITY = _get(_P, "recherche.seuil_doublons", float)
DEDUP_MIN_SHINGLES = _get(_P, "recherche.doublons_taille_min", int)
MAX_PASSAGES_PER_DOCUMENT = _get(_P, "recherche.max_passages_par_document", int)

# ============================================================================
# INTERFACE
# ============================================================================
APP_TITLE = _get(_P, "interface.titre", str)
APP_PAGE_TITLE = _get(_P, "interface.titre_onglet", str)
AVATAR_PATH = str(Path(__file__).resolve().parent / _get(_P, "interface.avatar", str))
MODE_DESCRIPTIONS = _get(_P, "interface.descriptions_modes", dict)
DEFAULT_WELCOME_MESSAGE = _get(_P, "interface.message_accueil", str)
STATUS_MESSAGES = {
    step: _get(_P, f"interface.etapes.{step}", str)
    for step in ("reformulation", "recherche", "tri", "redaction", "termine")
}
SCORE_GOOD = _get(_P, "interface.score_vert", float)
SCORE_MEDIUM = _get(_P, "interface.score_orange", float)

# ============================================================================
# JOURNALISATION
# ============================================================================
RAG_LOGS_DB = _get(_P, "base.journal_requetes", str)
LOG_DOCS_LIMIT = 10  # Limite docs dans logs JSON
VERBOSE = _get(_P, "interface.journal_detaille", bool)

# ============================================================================
# PROMPTS
# ============================================================================
NO_ANSWER_MESSAGE = _get(_P, "prompts.aucune_reponse", str).strip()
CONDENSE_PROMPT = _get(_P, "prompts.reformulation", str).strip()
check_template("prompts.reformulation", CONDENSE_PROMPT, {"history", "query"})

# Correspondance noms du fichier -> noms internes
_PROMPT_KEYS = {"systeme": "system", "reponse": "rag", "hyde": "hyde"}
_PROMPT_VARIABLES = {"system": set(), "rag": {"context", "query"}, "hyde": {"query"}}

PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {}
for _mode, _templates in _get(_P, "prompts.modes", dict).items():
    PROMPT_TEMPLATES[_mode] = {}
    for _key, _internal in _PROMPT_KEYS.items():
        _text = _get(_P, f"prompts.modes.{_mode}.{_key}", str).strip()
        check_template(f"prompts.modes.{_mode}.{_key}", _text, _PROMPT_VARIABLES[_internal])
        PROMPT_TEMPLATES[_mode][_internal] = _text

# Mode de réponse (administratif, technique, créatif)
PROMPT_MODE = os.getenv("PROMPT_MODE", _get(_P, "interface.mode_par_defaut", str))
if PROMPT_MODE not in PROMPT_TEMPLATES:
    raise ParametresError(
        f"Mode par défaut « {PROMPT_MODE} » inconnu ; modes définis : {list(PROMPT_TEMPLATES)}"
    )

# Prompt personnalisé (si défini dans l'environnement, remplace le mode)
CUSTOM_RAG_PROMPT = os.getenv("CUSTOM_RAG_PROMPT", None)
CUSTOM_HYDE_PROMPT = os.getenv("CUSTOM_HYDE_PROMPT", None)
CUSTOM_SYSTEM_PROMPT = os.getenv("CUSTOM_SYSTEM_PROMPT", None)

# ==================== FONCTIONS UTILITAIRES ====================


def get_prompt_template(prompt_type: str = "rag", mode: str = None) -> str:
    """
    Récupère le template de prompt selon le type et le mode.

    Args:
        prompt_type: 'rag', 'hyde', ou 'system'
        mode: 'administratif', 'technique', 'créatif' (ou None pour utiliser PROMPT_MODE)

    Returns:
        str: Template de prompt
    """
    # Utiliser prompt personnalisé si défini
    if prompt_type == "rag" and CUSTOM_RAG_PROMPT:
        return CUSTOM_RAG_PROMPT
    elif prompt_type == "hyde" and CUSTOM_HYDE_PROMPT:
        return CUSTOM_HYDE_PROMPT
    elif prompt_type == "system" and CUSTOM_SYSTEM_PROMPT:
        return CUSTOM_SYSTEM_PROMPT

    # Sinon utiliser le mode
    mode = mode or PROMPT_MODE

    if mode not in PROMPT_TEMPLATES:
        logging.getLogger(__name__).warning(
            "Mode '%s' inconnu, utilisation du mode '%s'", mode, list(PROMPT_TEMPLATES)[0]
        )
        mode = list(PROMPT_TEMPLATES)[0]

    return PROMPT_TEMPLATES[mode].get(prompt_type, "")


def set_prompt_mode(mode: str):
    """
    Change le mode de prompt.

    Args:
        mode: 'administratif', 'technique', ou 'créatif'
    """
    global PROMPT_MODE
    if mode in PROMPT_TEMPLATES:
        PROMPT_MODE = mode
        logging.getLogger(__name__).info("Mode de prompt changé : %s", mode)
    else:
        logging.getLogger(__name__).error(
            "Mode invalide. Modes disponibles : %s", list(PROMPT_TEMPLATES.keys())
        )


def list_prompt_modes() -> List[str]:
    """Liste tous les modes de prompts disponibles."""
    return list(PROMPT_TEMPLATES.keys())


def setup_logging() -> None:
    """
    Configure la journalisation des scripts et de l'application.

    Niveau INFO si VERBOSE, WARNING sinon ; messages bruts (sans préfixe)
    pour garder une sortie console lisible. Les bibliothèques HTTP, très
    bavardes au niveau INFO, sont limitées aux avertissements.
    """
    logging.basicConfig(
        level=logging.INFO if VERBOSE else logging.WARNING,
        format="%(message)s",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
