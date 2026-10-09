# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

Chat de Calais : assistant RAG (Streamlit) qui interroge les notes et documents de la DDTM du Pas-de-Calais (réunions bilatérales avec le préfet) via l'API Albert (Etalab), compatible OpenAI. Code, docstrings, messages et commits en français.

Choix arrêtés par l'utilisateur : dépendances gérées avec **pip et `requirements.txt` (pas de uv, pas de pyproject)** ; application Streamlit maison conservée (**pas d'Open WebUI**). Notes de reprise dans `memory.md`, métadonnées décrites dans `dictionnaire_donnees.md` : les tenir à jour quand on modifie des données ou métadonnées.

## Commandes

```bash
pip install -r requirements.txt -r requirements-dev.txt   # .venv local déjà présent : .venv/Scripts/python
cp .env.example .env                                       # ALBERT_API_KEY obligatoire

python main.py                    # indexation incrémentale de ./documents puis chat en console
streamlit run app_chat.py         # interface de chat
streamlit run app_logs.py         # tableau de bord des logs
python evaluate.py [--no-hyde] [--no-hybrid] [--no-rerank] [--no-dedup] [--max-per-doc N]
                   [--min-score X] [--seuils 0.005,0.03] [--base DOSSIER_CHROMA] [--generate]

pytest                                         # pytest.ini ajoute la couverture (écrit .coverage, coverage.xml, htmlcov)
pytest tests/test_corpus.py::test_cap_per_document --no-cov -q   # un seul test
```

Les tests n'appellent jamais Albert (API simulée par mocks ou `FakeListChatModel`). Les essais réels se font avec les scripts `_tmp/*.py` (dossier ignoré par git), qui pointent `config.CHROMA_DB_PATH` vers une base de test (`_tmp/chroma_full`, copie du cache OCR) pour ne pas toucher la base de l'application ; la variable d'environnement `CHROMA_DB_PATH` fait de même sans code. Le notebook `exemple_pipeline.ipynb` est généré et exécuté par `_tmp/build_notebook.py --executer` (textes dans `_tmp/notebook_textes.md`).

## Architecture

**Paramètres** : tout ce qui est réglable (modèles, URL, recherche, OCR, interface, prompts) est dans `parametres.yaml`, commenté pour un non-développeur. `config.py` le lit, le valide (`_get` vérifie type et présence, `check_template` les variables des prompts : `{context}` `{query}` pour la réponse, `{query}` pour HyDE, `{history}` `{query}` pour la reformulation) et expose les constantes historiques (`RAG_TOP_K_DOCS`, `PROMPT_TEMPLATES`...) : le reste du code n'importe que `config`. Les variables d'environnement priment (`.env`), `PARAMETRES_FILE` change de fichier. Une nouvelle option = une clé YAML + une constante dans `config.py` + mention dans `dictionnaire_donnees.md`. `setup_logging()` en tête de chaque point d'entrée (les modules utilisent `logging`, pas `print`).

**Chargement** (`loaders.load_document`, aiguillage par extension) : un `Document` LangChain par page (PDF) ou par fichier. OCR via Albert (`albert_client.ocr_image`, `openweight-ocr`, ou `/v1/ocr` `mistral-ocr-2512` avec repli) pour les pages PDF sans texte, les images insérées dans des pages de texte et les fichiers image ; cache brut dans `<base>/ocr_cache/<md5>_<moteur>.json`, nettoyé à la lecture par `normalize_ocr_text` (tableaux HTML en Markdown, boucles de répétition, consigne recopiée). `.doc` lu en Python pur par `doc_reader.py` ; `.eml` : corps puis pièces jointes (`source` = chemin du `.eml`, `attachment`) ; ODT sans pieds de page, commentaires ni texte supprimé du suivi des modifications.

**Corpus à documents similaires** (`corpus.py`) : le corpus contient brouillons et versions validées d'une même note, reprises d'une réunion à l'autre. `document_metadata` tire du chemin `meeting_date` (AAAAMMJJ du dossier « jj mm aaaa »), `validated`, `doc_title`, `doc_family`. `indexer.split_documents` les pose sur chaque chunk avec `chunk_index` et place un en-tête contextuel devant le texte (`header_len` ; `corpus.chunk_body` le retire : l'utiliser pour tout affichage ou comparaison de texte). À la recherche : `collapse_duplicates` (critère `near_duplicate` : Jaccard sur 5-grammes ou inclusion si le passage compte au moins `DEDUP_MIN_SHINGLES` empreintes ; représentant validé puis récent ; `aussi_dans`), `cap_per_document` (famille et date), filtres `build_where` (Chroma) et `matches_filters` (BM25).

**Indexation** (`indexer.index_documents`) : détection des changements par hash MD5 (`<base>/index_metadata.json`), suppression des chunks par `source` (y compris pour les nouveaux fichiers, reliquats d'un échec), fichiers exclus par `documents.exclure`. Un fichier sans contenu ou dont un lot échoue n'est pas inscrit et sera retenté. Changer le contenu ou les métadonnées des chunks impose une réindexation complète (vider la base en gardant `ocr_cache/` : une vingtaine de secondes).

**Recherche** (`retrieval.py`) : `make_retriever` renvoie un `HybridRetriever` (Chroma + BM25 en mémoire reconstruit au démarrage, fusion RRF) si `USE_HYBRID_SEARCH`. Typage canard avec extensions testées par `getattr` dans `rag_pipeline` :
- `supports_keyword_query is True` : `invoke(hyde_query, keyword_query=question, filters=...)` ;
- `get_neighbors(doc, window)` : chunks voisins ;
- `meeting_dates()` : bornes des filtres de période (`available_meeting_dates`).
`temp_documents.CombinedRetriever` (documents de session + base permanente) les implémente et délègue ; les filtres ne s'appliquent qu'à la base permanente.

**Pipeline** (`rag_pipeline.py`) : `iter_retrieval_steps` est un générateur d'événements `status` (`step`, `content` tiré de `interface.etapes`, `detail`, `elapsed`) qui enchaîne reformulation (`condense_question`), HyDE, recherche, dédoublonnage, rerank (copies avec `rerank_score`), seuil `RERANK_MIN_SCORE`, plafond, voisins et `format_context`, en remplissant `state`. `retrieve_passages` le consomme sans événements (utilisé par `evaluate.py` et le notebook) ; `rag_query_stream` le relaie puis streame la réponse et finit toujours par un `metadata` ; `rag_query` en est la version assemblée. Sans passage retenu, le LLM n'est pas appelé (`no_relevant_docs`, `NO_ANSWER_MESSAGE`).

**Citations** : `format_context` numérote les extraits `[n]` (source, date, statut) et `build_passages` décrit les mêmes passages dans le même ordre pour l'interface (`utils_app.render_passages` : onglets, extrait, aperçu de page PDF, téléchargement différé, « aussi dans »). Toute modification d'un côté doit préserver cette correspondance.

**Interface** (`app_chat.py`) : `st.status` + `st.spinner(show_time=True)` alimentés par les événements `status` ; filtres de période et de statut dans la barre latérale, passés en `filters` au pipeline. **Journalisation des requêtes** : `logger.RAGLogger` (SQLite) est alimenté par l'appelant (`app_chat.py`, `main.py`) à partir du `metadata` du flux ; `app_logs.py` et `view_logs.py` le lisent.

**Évaluation** : `evaluation/questions.yaml` (questions de référence et fragments de noms de fichiers attendus, `[]` pour les questions hors sujet), `evaluate.py` (variantes, simulation de seuils avec `--seuils`), résultats CSV dans `evaluation/resultats/` (ignoré par git), synthèse dans `evaluation/rapport_evaluation.md`.

## Points de vigilance

- Scores du reranker Albert bas et très contrastés (pertinent 0,4 à 0,99, hors sujet proche de 0) : seuils `recherche.score_minimum`, `recherche.score_alerte`, `interface.score_vert`, `interface.score_orange` calibrés avec `evaluate.py` (voir le rapport).
- `config.ParametresError` est conservée lors d'un `importlib.reload(config)` (tests) ; dans les tests, référencer `config.X` au moment de l'appel plutôt qu'importer les noms.
- Streamlit >= 1.52 requis (`st.download_button` avec données différées, `st.spinner(show_time=True)`).
- Sous Windows, Chroma garde ses fichiers ouverts : la fixture `temp_dir` de `tests/conftest.py` vide le cache des clients Chroma avant suppression ; ne pas réécrire une base ouverte par l'application en cours.
- `documents/` et `chroma_db_rag/` contiennent des données réelles non versionnées : pas de réindexation complète sans raison (quota Albert ; l'OCR est en cache).
