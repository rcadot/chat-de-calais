# app_chat.py (VERSION FINALE AVEC FEEDBACK AU PREMIER CLIC)
"""Application Streamlit de chat RAG avec modes de prompt."""

import streamlit as st
from datetime import datetime
from pathlib import Path
import time
import config
from albert_client import get_embeddings, get_llm
from indexer import index_documents
from rag_pipeline import rag_query, rag_query_stream
from logger import RAGLogger
from temp_documents import render_temp_documents_section
from st_copy import copy_button
from utils_app import format_sources, render_passages
from corpus import format_meeting_date
from retrieval import available_meeting_dates

config.setup_logging()

# Avatar de l'assistant (parametres.yaml : interface.avatar), emoji à défaut
AVATAR = config.AVATAR_PATH if Path(config.AVATAR_PATH).is_file() else "🐈"

# ✅ MODE DEV : Changer à False pour utiliser l'IA réelle
DEV_MODE = False

# Configuration de la page
st.set_page_config(
    page_title=config.APP_PAGE_TITLE,
    page_icon=AVATAR,
    layout="wide",
    initial_sidebar_state="expanded",
)
if AVATAR != "🐈":
    st.logo(AVATAR, size="large")

# CSS personnalisé
st.markdown(
    """
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .sources-section {
        margin-top: 1rem;
        padding: 0.75rem;
        background-color: #f8f9fa;
        border-left: 3px solid #0066cc;
        border-radius: 0.3rem;
    }
    .source-item {
        margin: 0.3rem 0;
        padding: 0.3rem 0.5rem;
        background-color: white;
        border-radius: 0.2rem;
        font-size: 0.9rem;
    }
    .source-score {
        display: inline-block;
        padding: 0.1rem 0.4rem;
        background-color: #28a745;
        color: white;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .source-score-medium {
        background-color: #ffc107;
    }
    .source-score-low {
        background-color: #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialisation du système
@st.cache_resource
def init_rag_system():
    """Initialise le système RAG (une seule fois)."""
    with st.spinner("🔄 Initialisation du système RAG..."):
        embeddings = get_embeddings()
        llm = get_llm()
        logger = RAGLogger(config.RAG_LOGS_DB)
        vectorstore, retriever = index_documents(embeddings)
    return retriever, llm, logger


# Initialiser le state de session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_initialized" not in st.session_state:
    try:
        retriever, llm, logger = init_rag_system()
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.logger = logger
        st.session_state.system_initialized = True
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation : {e}")
        st.stop()

# ✅ Initialiser le tracking des feedbacks
if "feedbacks_given" not in st.session_state:
    st.session_state.feedbacks_given = set()

# ✅ NOUVEAU : Gestion du feedback en attente
if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = None

# ✅ NOUVEAU : Traiter le feedback en attente au début du script
if st.session_state.pending_feedback:
    query_id, feedback_type = st.session_state.pending_feedback
    if st.session_state.logger.update_feedback(query_id, feedback_type):
        st.session_state.feedbacks_given.add(f"feedback_{query_id}")
        emoji = "👍" if feedback_type == "thumbs_up" else "👎"
        st.toast(f"✅ Merci pour votre retour !", icon=emoji)
    st.session_state.pending_feedback = None
    st.rerun()

# Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    # Mode de prompt
    st.subheader("🎭 Mode de réponse")
    modes = config.list_prompt_modes()
    prompt_mode = st.selectbox(
        "Sélectionnez le style",
        modes,
        index=modes.index(config.PROMPT_MODE) if config.PROMPT_MODE in modes else 0,
        help="Change le style et le ton des réponses",
    )

    # Description du mode (parametres.yaml : interface.descriptions_modes)
    if config.MODE_DESCRIPTIONS.get(prompt_mode):
        st.info(config.MODE_DESCRIPTIONS[prompt_mode])

    st.divider()

    # Filtres de recherche (documents similaires : période, versions validées)
    st.subheader("🔎 Filtres")
    if "meeting_dates" not in st.session_state:
        try:
            st.session_state.meeting_dates = available_meeting_dates(st.session_state.retriever)
        except Exception:
            st.session_state.meeting_dates = []
    meeting_dates = st.session_state.meeting_dates
    search_filters = {}
    # Filtre de période désactivé par défaut : un document récemment indexé
    # ou sans date de réunion serait sinon écarté sans que l'utilisateur le sache.
    if len(meeting_dates) >= 2 and st.toggle(
        "Filtrer par période",
        value=False,
        help="Désactivé : tous les documents sont interrogés, quelle que soit leur date",
    ):
        date_min, date_max = st.select_slider(
            "Période des réunions",
            options=meeting_dates,
            value=(meeting_dates[0], meeting_dates[-1]),
            format_func=format_meeting_date,
            help="Ne chercher que dans les documents des réunions de cette période",
        )
        if (date_min, date_max) != (meeting_dates[0], meeting_dates[-1]):
            search_filters.update(date_min=date_min, date_max=date_max)
    if st.checkbox(
        "Versions validées uniquement",
        help="Ignorer les brouillons : ne garder que les versions validées, finales ou définitives",
    ):
        search_filters["validated_only"] = True
    if search_filters:
        st.caption("Filtres actifs : les documents temporaires restent toujours interrogés.")

    st.divider()

    render_temp_documents_section(get_embeddings())

    # Actions
    if st.button("🗑️ Effacer conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedbacks_given = set()
        st.session_state.pending_feedback = None
        st.rerun()

# En-tête principal
st.title(config.APP_TITLE)
st.caption(f"Posez vos questions sur les documents - Mode: **{prompt_mode}**")

if len(st.session_state.messages) == 0:
    welcome_msg = config.DEFAULT_WELCOME_MESSAGE
    st.info(welcome_msg)

# Afficher l'historique des messages
for idx, message in enumerate(st.session_state.messages):
    avatar = AVATAR if message["role"] == "assistant" else None

    with st.chat_message(message["role"], avatar=avatar):
        # Afficher le contenu
        st.markdown(message["content"])

        # Pour les messages assistant : bouton copier + feedback + sources collapsées
        if message["role"] == "assistant":
            # Créer des colonnes pour copier et feedback
            col1, col2, col3 = st.columns([0.92, 0.04, 0.04])

            with col1:
                # Bouton copier
                copy_button(
                    text=message["content"],
                    tooltip="Copier cette réponse",
                    copied_label="Copié !",
                    key=f"copy_{idx}",
                )

            # Vérifier si feedback déjà donné
            query_id = message.get("query_id")
            feedback_key = f"feedback_{query_id}"
            already_voted = feedback_key in st.session_state.feedbacks_given

            with col2:
                # ✅ Bouton pouce en l'air - utiliser on_click
                st.button(
                    "👍",
                    key=f"thumbs_up_{idx}",
                    help="Bonne réponse" if not already_voted else "Déjà évalué",
                    disabled=already_voted,
                    on_click=lambda qid=query_id: setattr(
                        st.session_state, "pending_feedback", (qid, "thumbs_up")
                    ),
                )

            with col3:
                # ✅ Bouton pouce en bas - utiliser on_click
                st.button(
                    "👎",
                    key=f"thumbs_down_{idx}",
                    help="Mauvaise réponse" if not already_voted else "Déjà évalué",
                    disabled=already_voted,
                    on_click=lambda qid=query_id: setattr(
                        st.session_state, "pending_feedback", (qid, "thumbs_down")
                    ),
                )

            # Sources (SOUS les boutons)
            if "passages" in message:
                render_passages(
                    message["passages"],
                    key_prefix=f"msg_{idx}",
                    no_relevant_docs=message.get("no_relevant_docs", False),
                )
            elif "sources" in message and message["sources"]:
                with st.expander("📚 Voir les sources consultées", expanded=False):
                    sources_html = format_sources(
                        message["sources"], message.get("scores", [])
                    )
                    st.markdown(sources_html, unsafe_allow_html=True)

# Zone de saisie
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer la réponse
    with st.chat_message("assistant", avatar=AVATAR):
        # Statut animé (étapes et détails), puis la réponse en dessous
        status_box = st.status(config.STATUS_MESSAGES["recherche"], expanded=True)
        message_placeholder = st.empty()

        try:
            # ✅ MODE DEV : Réponse simulée
            if DEV_MODE:
                dev_response = f"🔧 **[MODE DEV]** Vous avez demandé : *{prompt}*\n\nCeci est une réponse de test sans appel à l'IA."

                # Simuler le streaming
                full_response = ""
                for char in dev_response:
                    full_response += char
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)

                message_placeholder.markdown(full_response)
                metadata = {
                    "sources": [],
                    "rerank_scores": [],
                    "hyde_query": None,
                    "retrieved_docs": None,
                    "reranked_docs": None,
                    "execution_time": 0,
                    "error": None,
                }
                query_id = None  # Pas de logging en mode DEV

            # ✅ MODE NORMAL : RAG complet
            else:
                full_response = ""
                metadata = None
                with status_box, st.spinner("Temps écoulé", show_time=True):
                    for item in rag_query_stream(
                        prompt,
                        st.session_state.retriever,
                        st.session_state.llm,
                        logger=None,  # Ne pas logger dans le pipeline
                        mode=prompt_mode,
                        # Historique sans la question en cours (questions de suivi)
                        history=st.session_state.messages[:-1],
                        filters=search_filters or None,
                    ):
                        if item["type"] == "status":
                            if item.get("detail"):
                                st.markdown(f"✓ {item['detail']}  \n:gray[{item['elapsed']:.1f} s]")
                            if item.get("step") == "redaction":
                                status_box.update(
                                    label=f"{config.STATUS_MESSAGES['termine']} en {item['elapsed']:.1f} s",
                                    state="complete",
                                    expanded=False,
                                )
                                message_placeholder.markdown(f"_{item['content']}_")
                            else:
                                status_box.update(label=item["content"])
                        elif item["type"] == "chunk":
                            full_response += item["content"]
                            message_placeholder.markdown(full_response + "▌")
                        elif item["type"] == "metadata":
                            metadata = item
                        elif item["type"] == "error":
                            st.error(item["content"])
                            full_response = item["content"]
                            metadata = {
                                "sources": [],
                                "rerank_scores": [],
                                "error": str(item["content"]),
                                "hyde_query": None,
                                "retrieved_docs": None,
                                "reranked_docs": None,
                                "execution_time": 0,
                            }

                # Retirer le curseur ; statut final si la rédaction n'a pas eu lieu
                if full_response:
                    message_placeholder.markdown(full_response)
                if metadata and metadata.get("error"):
                    status_box.update(label="Erreur pendant le traitement", state="error")
                elif metadata and metadata.get("no_relevant_docs"):
                    status_box.update(
                        label="Aucune source suffisamment pertinente", state="complete", expanded=False
                    )
                if metadata and metadata.get("execution_time"):
                    st.caption(f"Réponse en {metadata['execution_time']:.1f} s")

                # ✅ Logger UNE SEULE FOIS avec TOUTES les infos du pipeline
                if metadata:
                    query_id = st.session_state.logger.log_query(
                        user_query=prompt,
                        hyde_query=metadata.get("hyde_query"),
                        retrieved_docs=metadata.get("retrieved_docs"),
                        reranked_docs=metadata.get("reranked_docs"),
                        final_answer=full_response,
                        sources=metadata.get("sources", []),
                        rerank_scores=metadata.get("rerank_scores", []),
                        execution_time=metadata.get("execution_time", 0),
                        error=metadata.get("error"),
                        prompt_mode=prompt_mode,
                    )
                else:
                    # Fallback si pas de metadata
                    query_id = st.session_state.logger.log_query(
                        user_query=prompt,
                        hyde_query=None,
                        retrieved_docs=None,
                        reranked_docs=None,
                        final_answer=full_response,
                        sources=[],
                        rerank_scores=[],
                        execution_time=0,
                        error="Pas de metadata reçue",
                        prompt_mode=prompt_mode,
                    )

            # Boutons copier et feedback
            col1, col2, col3 = st.columns([0.92, 0.04, 0.04])

            with col1:
                copy_button(
                    text=full_response,
                    tooltip="Copier cette réponse",
                    copied_label="Copié !",
                    key="copy_current",
                )

            with col2:
                # ✅ Utiliser on_click pour le feedback immédiat
                st.button(
                    "👍",
                    key="thumbs_up_current",
                    help="Bonne réponse",
                    on_click=lambda: setattr(
                        st.session_state, "pending_feedback", (query_id, "thumbs_up")
                    ),
                )

            with col3:
                # ✅ Utiliser on_click pour le feedback immédiat
                st.button(
                    "👎",
                    key="thumbs_down_current",
                    help="Mauvaise réponse",
                    on_click=lambda: setattr(
                        st.session_state, "pending_feedback", (query_id, "thumbs_down")
                    ),
                )

            # Sources détaillées (numérotées comme les citations de la réponse)
            if metadata:
                render_passages(
                    metadata.get("passages", []),
                    key_prefix=f"msg_{len(st.session_state.messages)}",
                    no_relevant_docs=metadata.get("no_relevant_docs", False),
                )

            # Ajouter à l'historique
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": metadata.get("sources", []) if metadata else [],
                    "scores": metadata.get("rerank_scores", []) if metadata else [],
                    "passages": metadata.get("passages", []) if metadata else [],
                    "no_relevant_docs": (
                        metadata.get("no_relevant_docs", False) if metadata else False
                    ),
                    "query_id": query_id,
                }
            )

        except Exception as e:
            message_placeholder.empty()
            st.error(f"❌ Erreur inattendue : {e}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Désolé, une erreur s'est produite : {e}",
                }
            )
