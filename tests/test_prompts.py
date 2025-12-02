# tests/test_prompts.py
"""Tests des modes de prompts."""
import pytest
import config

def test_list_prompt_modes():
    """Test liste des modes."""
    modes = config.list_prompt_modes()
    assert "administratif" in modes
    assert "technique" in modes
    assert "créatif" in modes

def test_get_prompt_template_default():
    """Test récupération template par défaut."""
    template = config.get_prompt_template("rag")
    assert "{context}" in template
    assert "{query}" in template

def test_get_prompt_template_specific_mode():
    """Test récupération template mode spécifique."""
    template = config.get_prompt_template("rag", mode="technique")
    assert "technique" in template.lower()

def test_set_prompt_mode():
    """Test changement de mode."""
    original = config.PROMPT_MODE
    config.set_prompt_mode("créatif")
    assert config.PROMPT_MODE == "créatif"
    config.PROMPT_MODE = original  # Restore

def test_custom_prompt_override(monkeypatch):
    """Test que custom prompt override le mode."""
    custom = "Custom prompt {context} {query}"
    monkeypatch.setattr(config, "CUSTOM_RAG_PROMPT", custom)
    
    template = config.get_prompt_template("rag")
    assert template == custom
