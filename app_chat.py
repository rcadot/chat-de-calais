# app_chat.py (VERSION FINALE AVEC SPINNER)
"""Application Streamlit de chat RAG avec modes de prompt."""
import streamlit as st
from datetime import datetime
from pathlib import Path
import config
from albert_client import get_embeddings, get_llm
from indexer import index_documents
from rag_pipeline import rag_query, rag_query_stream
from logger import RAGLogger
from temp_documents import render_temp_documents_section

# Configuration de la page
st.set_page_config(
    page_title="Chat RAG Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS personnalisé
st.markdown("""
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
    /* ✅ NOUVEAU : Style pour le bouton copier */
    .copy-button {
        position: relative;
        float: right;
        margin-top: -2rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


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


# Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Mode de prompt
    st.subheader("🎭 Mode de réponse")
    prompt_mode = st.selectbox(
        "Sélectionnez le style",
        ["administratif", "technique", "créatif"],
        index=0,
        help="Change le style et le ton des réponses"
    )
    
    # Description du mode
    mode_descriptions = {
        "administratif": "📋 Réponses formelles et réglementaires",
        "technique": "🔧 Réponses détaillées avec procédures",
        "créatif": "💡 Réponses pédagogiques et accessibles"
    }
    st.info(mode_descriptions[prompt_mode])
    
    st.divider()

    render_temp_documents_section(get_embeddings())
    
    # Actions
    if st.button("🗑️ Effacer conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    


# En-tête principal
st.title("💬 Assistant RAG Conversationnel")
st.caption(f"Posez vos questions sur les documents - Mode: **{prompt_mode}**")

if len(st.session_state.messages) == 0:
    welcome_msg = config.DEFAULT_WELCOME_MESSAGE
    st.info(welcome_msg)
    

# Fonction pour formater les sources
def format_sources(sources, scores):
    """
    Formate les sources avec scores en HTML.
    Déduplique les sources et garde le meilleur score.
    """
    if not sources:
        return ""
    
    # Dédupliquer : garder le meilleur score pour chaque source unique
    sources_dict = {}
    for source, score in zip(sources, scores):
        source_name = Path(source).name
        
        if source_name not in sources_dict:
            sources_dict[source_name] = score
        else:
            # Garder le meilleur score
            if score and sources_dict[source_name]:
                sources_dict[source_name] = max(sources_dict[source_name], score)
            elif score:
                sources_dict[source_name] = score
    
    # Trier par score décroissant
    sorted_sources = sorted(
        sources_dict.items(), 
        key=lambda x: x[1] if x[1] else 0, 
        reverse=True
    )
    
    sources_html = '<div class="sources-section">'
    sources_html += '<strong>📚 Sources consultées :</strong><br><br>'
    
    for i, (source_name, score) in enumerate(sorted_sources, 1):
        if score:
            percentage = score * 100
            if score >= 0.7:
                score_class = "source-score"
                emoji = "🟢"
            elif score >= 0.5:
                score_class = "source-score source-score-medium"
                emoji = "🟡"
            else:
                score_class = "source-score source-score-low"
                emoji = "🔴"
            score_display = f'<span class="{score_class}">{emoji} {percentage:.0f}%</span>'
        else:
            score_display = '<span class="source-score">⚪ N/A</span>'
        
        sources_html += f'<div class="source-item">{score_display} <code>{source_name}</code></div>'
    
    sources_html += '</div>'
    return sources_html


# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Afficher les sources intégrées pour l'assistant
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            sources_html = format_sources(message["sources"], message.get("scores", []))
            st.markdown(sources_html, unsafe_allow_html=True)


# Zone de saisie
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Générer la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        # Emojis par mode
        mode_emojis = {
            "administratif": "📋",
            "technique": "🔧", 
            "créatif": "💡"
        }
        
        try:
            # ✅ Spinner Streamlit simple
            with st.spinner(f"{mode_emojis[prompt_mode]} Génération de la réponse..."):
                full_response = ""
                metadata = None
                
                for item in rag_query_stream(
                    prompt,
                    st.session_state.retriever,
                    st.session_state.llm,
                    logger=st.session_state.logger,
                    mode=prompt_mode
                ):
                    if item["type"] == "chunk":
                        full_response += item["content"]
                        message_placeholder.markdown(full_response + "▌")
                    elif item["type"] == "metadata":
                        metadata = item
                    elif item["type"] == "error":
                        st.error(item["content"])
                        full_response = item["content"]
                        metadata = {"sources": [], "rerank_scores": []}
            
            # Retirer le curseur
            if full_response:
                message_placeholder.markdown(full_response)
            
            # Afficher les sources
            if metadata and metadata.get("sources"):
                sources_html = format_sources(metadata["sources"], metadata["rerank_scores"])
                sources_placeholder.markdown(sources_html, unsafe_allow_html=True)
                
                # Ajouter à l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": metadata["sources"],
                    "scores": metadata["rerank_scores"]
                })
            else:
                # Pas de metadata (erreur)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": [],
                    "scores": []
                })
            
        except Exception as e:
            message_placeholder.empty()
            st.error(f"❌ Erreur inattendue : {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Désolé, une erreur s'est produite : {e}",
            })


