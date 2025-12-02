# loaders.py
"""Loaders pour différents formats de documents."""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
    UnstructuredMarkdownLoader, UnstructuredHTMLLoader
)
import config


def load_odt(file_path: str) -> List[Document]:
    """Charge fichier ODT."""
    try:
        from odf import text, teletype
        from odf.opendocument import load
        
        textdoc = load(file_path)
        allparas = textdoc.getElementsByType(text.P)
        content = "\n".join([teletype.extractText(para) for para in allparas])
        
        return [Document(page_content=content, metadata={"source": file_path})]
    except Exception as e:
        if config.VERBOSE:
            print(f"⚠️ Erreur ODT {Path(file_path).name}: {e}")
        return []


def load_document(file_path: str) -> List[Document]:
    """Charge un document selon son extension."""
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext == '.pdf':
            return PyPDFLoader(file_path).load()
        
        elif ext == '.docx':
            return Docx2txtLoader(file_path).load()
        
        elif ext in ['.txt', '.text']:
            for encoding in config.TEXT_ENCODINGS:
                try:
                    return TextLoader(file_path, encoding=encoding).load()
                except:
                    continue
            return []
        
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path).load()
        
        elif ext in ['.html', '.htm']:
            return UnstructuredHTMLLoader(file_path).load()
        
        elif ext == '.odt':
            return load_odt(file_path)
        
        elif ext == '.doc':
            if config.VERBOSE:
                print(f"⏭️ .doc ignoré: {Path(file_path).name}")
            return []
        
        return []
        
    except Exception as e:
        if config.VERBOSE:
            print(f"⚠️ Erreur {Path(file_path).name}: {e}")
        return []
