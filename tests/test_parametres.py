# tests/test_parametres.py
"""Tests du fichier de paramètres (parametres.yaml) et de sa validation."""

import importlib
import os
from pathlib import Path

import pytest
import yaml

import config


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def reload_with(monkeypatch, temp_dir):
    """Recharge config avec un fichier de paramètres modifié, puis restaure."""

    def _reload(modify) -> None:
        data = yaml.safe_load((ROOT / "parametres.yaml").read_text(encoding="utf-8"))
        modify(data)
        path = os.path.join(temp_dir, "parametres.yaml")
        Path(path).write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        monkeypatch.setenv("PARAMETRES_FILE", path)
        importlib.reload(config)

    yield _reload
    monkeypatch.delenv("PARAMETRES_FILE", raising=False)
    importlib.reload(config)


def test_repository_file_is_valid():
    """Le fichier livré se charge et définit tous les modes et étapes."""
    params = config.load_parameters(str(ROOT / "parametres.yaml"))
    assert set(params["prompts"]["modes"]) == {"administratif", "technique", "créatif"}
    assert config.PROMPT_MODE in config.PROMPT_TEMPLATES
    assert set(config.STATUS_MESSAGES) == {"reformulation", "recherche", "tri", "redaction", "termine"}


def test_values_reach_constants(reload_with):
    """Une valeur modifiée dans le fichier se retrouve dans la constante."""
    reload_with(lambda d: d["recherche"].update(passages_retenus=7, score_minimum=0.05))
    assert config.RAG_TOP_K_DOCS == 7
    assert config.RERANK_MIN_SCORE == 0.05


def test_env_overrides_file(reload_with, monkeypatch):
    """Une variable d'environnement prime sur le fichier."""
    monkeypatch.setenv("USE_HYBRID_SEARCH", "false")
    reload_with(lambda d: d["recherche"].update(hybride=True))
    assert config.USE_HYBRID_SEARCH is False


def test_missing_key_message():
    """Clé absente : message qui nomme la clé."""
    with pytest.raises(config.ParametresError, match="recherche.inexistante"):
        config._get({"recherche": {}}, "recherche.inexistante", int)


def test_wrong_type_message():
    """Mauvais type : message explicite avec la valeur lue."""
    with pytest.raises(config.ParametresError, match="doit être un nombre entier.*'cinq'"):
        config._get({"recherche": {"passages_retenus": "cinq"}}, "recherche.passages_retenus", int)
    with pytest.raises(config.ParametresError, match="true ou false"):
        config._get({"recherche": {"hyde": "oui"}}, "recherche.hyde", bool)
    assert config._get({"a": {"b": 1}}, "a.b", float) == 1.0  # entier accepté pour un nombre


def test_template_variables_checked():
    """Prompt sans {context}, ou avec une variable inconnue : erreur claire."""
    with pytest.raises(config.ParametresError, match=r"absente.*\{context\}"):
        config.check_template("reponse", "Question : {query}", {"context", "query"})
    with pytest.raises(config.ParametresError, match=r"inconnue.*\{contexte\}"):
        config.check_template("reponse", "{contexte} {query} {context}", {"context", "query"})
    config.check_template("reponse", "{context} {query} accolade {{littérale}}", {"context", "query"})


def test_invalid_prompt_stops_loading(reload_with):
    """Un prompt invalide dans le fichier empêche le démarrage."""
    with pytest.raises(config.ParametresError, match="prompts.modes.technique.reponse"):
        reload_with(lambda d: d["prompts"]["modes"]["technique"].update(reponse="Sans contexte {query}"))


def test_yaml_syntax_error_gives_line(temp_dir):
    """Erreur de syntaxe YAML : numéro de ligne indiqué."""
    path = os.path.join(temp_dir, "casse.yaml")
    Path(path).write_text("recherche:\n  hyde: true\n   rerank: false\n", encoding="utf-8")
    with pytest.raises(config.ParametresError, match="ligne 3"):
        config.load_parameters(path)


def test_avatar_is_valid_image():
    """L'avatar configuré existe et est une image carrée."""
    from PIL import Image

    with Image.open(config.AVATAR_PATH) as image:
        assert image.size[0] == image.size[1] >= 64
