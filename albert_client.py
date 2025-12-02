# albert_client.py
"""Client Albert API simplifié."""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import config


def get_embeddings():
    """Retourne le client embeddings."""
    return OpenAIEmbeddings(
        model=config.EMBEDDINGS_MODEL,
        openai_api_key=config.ALBERT_API_KEY,
        openai_api_base=config.ALBERT_BASE_URL,
        model_kwargs={"encoding_format": config.EMBEDDINGS_CONFIG["encoding_format"]},
        chunk_size=config.EMBEDDINGS_CONFIG["chunk_size"],
        max_retries=config.EMBEDDINGS_CONFIG["max_retries"],
        request_timeout=config.EMBEDDINGS_CONFIG["request_timeout"]
    )


def get_llm():
    """Retourne le client LLM."""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_key=config.ALBERT_API_KEY,
        openai_api_base=config.ALBERT_BASE_URL,
        temperature=config.LLM_TEMPERATURE
    )
