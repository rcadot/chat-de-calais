# albert_client.py
"""Client Albert API simplifié."""
import base64
from typing import Dict, List, Optional

import requests
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
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


def ocr_image(image_bytes: bytes, mime: str = "image/png") -> str:
    """
    Transcrit une image de page via le modèle OCR ouvert d'Albert (/chat/completions).

    Args:
        image_bytes: Contenu binaire de l'image
        mime: Type MIME de l'image

    Returns:
        str: Texte transcrit (Markdown)
    """
    client = OpenAI(
        api_key=config.ALBERT_API_KEY,
        base_url=config.ALBERT_BASE_URL,
        timeout=config.OCR_TIMEOUT,
    )
    b64 = base64.b64encode(image_bytes).decode("ascii")
    response = client.chat.completions.create(
        model=config.OCR_MODELS["openweight"],
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def ocr_document_mistral(
    data: bytes, mime: str = "application/pdf", pages: Optional[List[int]] = None
) -> Dict[int, str]:
    """
    Transcrit un document via l'endpoint /ocr d'Albert (Mistral OCR, accès restreint).

    Args:
        data: Contenu binaire du PDF ou de l'image
        mime: Type MIME du document
        pages: Indices de pages à traiter (base 0), None pour toutes

    Returns:
        Dict[int, str]: Texte Markdown par indice de page (base 0)

    Raises:
        requests.HTTPError: Si l'API refuse la requête (droits, taille...)
    """
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    if mime.startswith("image/"):
        document = {"type": "image_url", "image_url": data_url}
    else:
        document = {"type": "document_url", "document_url": data_url}

    payload = {"model": config.OCR_MODELS["mistral"], "document": document}
    if pages is not None:
        payload["pages"] = pages

    response = requests.post(
        f"{config.ALBERT_BASE_URL}/ocr",
        json=payload,
        headers={"Authorization": f"Bearer {config.ALBERT_API_KEY}"},
        timeout=config.OCR_TIMEOUT,
    )
    response.raise_for_status()

    result = {}
    for position, page in enumerate(response.json().get("pages", [])):
        index = page.get("index", pages[position] if pages else position)
        result[int(index)] = page.get("markdown", "")
    return result
