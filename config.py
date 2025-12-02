# config.py
"""Configuration simple pour le système RAG."""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API
# ============================================================================
ALBERT_API_KEY = os.getenv('ALBERT_API_KEY', '')
ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"

# ============================================================================
# MODÈLES
# ============================================================================
EMBEDDINGS_MODEL = "embeddings-small"
LLM_MODEL = "albert-large"
RERANK_MODEL = "rerank-small"

# ============================================================================
# EMBEDDINGS
# ============================================================================
EMBEDDINGS_CONFIG = {
    "encoding_format": "float",
    "chunk_size": 50,
    "max_retries": 3,
    "request_timeout": 60
}

# ============================================================================
# LLM
# ============================================================================
LLM_TEMPERATURE = 0.1

# ============================================================================
# CHUNKING
# ============================================================================
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

# ============================================================================
# INDEXATION
# ============================================================================
DOCUMENTS_DIR = "./documents"
CHROMA_DB_PATH = "./chroma_db_rag"
COLLECTION_NAME = "documents_rag"
BATCH_SIZE = 50
RETRIEVER_TOP_K = 30

SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.odt', '.txt', '.md', '.html', '.htm']
TEXT_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

# ============================================================================
# RAG PIPELINE
# ============================================================================
RAG_TOP_K_DOCS = 5
RAG_TOP_N_RETRIEVAL = 30
USE_HYDE = True
USE_RERANK = True

# ============================================================================
# LOGGING
# ============================================================================
RAG_LOGS_DB = "./rag_logs.db"
LOG_DOCS_LIMIT = 10  # Limite docs dans logs JSON
VERBOSE = True


# ==================== CONFIGURATION DES PROMPTS ====================

# Mode de réponse (administratif, technique, créatif)
PROMPT_MODE = os.getenv("PROMPT_MODE", "administratif")  # administratif | technique | créatif

# Templates de prompts par mode
PROMPT_TEMPLATES = {
    "administratif": {
        "system": """Tu es un assistant IA spécialisé dans l'aide aux administrations publiques françaises.
Tu dois fournir des réponses précises, factuelles et conformes aux réglementations en vigueur.
Utilise un ton formel et professionnel. Cite systématiquement tes sources.""",
        
        "rag": """Contexte réglementaire et documentaire :
{context}

Question de l'agent public :
{query}

Instructions :
- Réponds de manière claire et structurée
- Cite les documents sources entre crochets [nom_du_document]
- Mentionne les articles ou références réglementaires pertinents
- Si l'information n'est pas dans le contexte, indique-le clairement
- Utilise un vocabulaire administratif précis

Réponse :""",

        "hyde": """En tant qu'expert en administration publique française, rédige un document de référence détaillé qui répondrait à la question suivante :

Question : {query}

Document de référence (format administratif formel) :"""
    },
    
    "technique": {
        "system": """Tu es un assistant technique spécialisé dans l'analyse de documents et procédures.
Tu dois fournir des réponses détaillées avec des explications techniques précises.
Structure tes réponses avec des listes, étapes et exemples concrets.""",
        
        "rag": """Documentation technique disponible :
{context}

Question technique :
{query}

Instructions :
- Fournis une réponse technique détaillée
- Décompose en étapes numérotées si nécessaire
- Ajoute des exemples ou cas d'usage
- Explique les concepts complexes simplement
- Cite les sources [nom_du_document]

Réponse technique :""",

        "hyde": """En tant qu'expert technique, crée une documentation détaillée avec procédures et exemples pour répondre à cette question :

Question : {query}

Documentation technique complète :"""
    },
    
    "créatif": {
        "system": """Tu es un assistant créatif et pédagogique qui rend l'information accessible et engageante.
Tu utilises des analogies, des exemples concrets et un ton convivial tout en restant précis.""",
        
        "rag": """Informations disponibles :
{context}

Question :
{query}

Instructions :
- Explique de manière claire et accessible
- Utilise des analogies ou exemples parlants si pertinent
- Structure ta réponse de façon logique
- Reste factuel malgré le ton convivial
- Mentionne tes sources [nom_du_document]

Réponse :""",

        "hyde": """Imagine une explication détaillée et pédagogique qui répondrait parfaitement à cette question :

Question : {query}

Explication complète et accessible :"""
    }
}

# Prompt personnalisé (si défini, remplace le mode)
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
        if VERBOSE:
            print(f"⚠️ Mode '{mode}' inconnu, utilisation du mode 'administratif'")
        mode = "administratif"
    
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
        if VERBOSE:
            print(f"✅ Mode de prompt changé : {mode}")
    else:
        print(f"❌ Mode invalide. Modes disponibles : {list(PROMPT_TEMPLATES.keys())}")

def list_prompt_modes():
    """Liste tous les modes de prompts disponibles."""
    return list(PROMPT_TEMPLATES.keys())


DEFAULT_WELCOME_MESSAGE = """
👋 **Bienvenue dans l'Assistant RAG**

Posez vos questions sur les documents disponibles dans la base de connaissances.

💡 Changez le mode de réponse dans la barre latérale pour adapter le style.
"""