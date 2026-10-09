# Mémoire de travail

## 2026-10-09 : OCR des PDF scannés et corrections du pipeline

Plan suivi : options A1 (OCR) et A2 (corrections) ; A3 (BM25, Docling, uv) et le POC Open WebUI restent à faire.

### Fait
- `loaders.load_pdf` : PyPDF page par page, OCR des pages ayant moins de `OCR_MIN_CHARS` caractères utiles.
  - Moteur `openweight` : page rendue en PNG (pypdfium2, 200 DPI, côté max 1540 px) puis `/chat/completions` modèle `openweight-ocr`.
  - Moteur `mistral` : `/v1/ocr` modèle `mistral-ocr-2512` (accès restreint) avec repli automatique sur `openweight`.
  - Cache : `chroma_db_rag/ocr_cache/<md5>_<moteur>.json` (texte par page).
- Images (`.png .jpg .jpeg .tif .tiff`) indexables via `loaders.load_image`.
- `indexer` : un fichier sans chunk n'est plus inscrit dans `index_metadata.json` (retenté au prochain lancement).
- `rag_pipeline.format_context` : chaque extrait est préfixé par `[fichier, p. N]` pour permettre les citations.
- HyDE reçoit le mode choisi ; rerank sur `RERANK_MAX_CHARS` (= `CHUNK_SIZE`) ; clés de métadonnées de `rag_query_stream` alignées sur `rag_query` (`execution_time`, `n_docs_*`, `prompt_mode`).
- Documents temporaires : découpés avec `indexer.get_text_splitter`, collection Chroma recréée à chaque mise à jour (fin des doublons), interrogés avec la base permanente via `CombinedRetriever` (case « Interroger uniquement ces documents »).

### Fait le 2026-10-09 (2e passe)
- PDF mixtes : `loaders.find_image_regions` repère les images couvrant 10 à 85 % d'une page de texte, `ocr_image_regions` les découpe et transcrit (texte ajouté sous `[Texte extrait d'une image]`, métadonnée `ocr_images`). Testé en réel : OK.
- Recherche hybride : `retrieval.py` (BM25 `rank_bm25` + vectoriel, fusion RRF k=60). La question d'origine sert à BM25, la requête HyDE au vectoriel (`rag_pipeline.retrieve`).
- Prompts : citation au format `[nom_du_document, p. N]`, aligné sur les en-têtes du contexte.
- `.env.example` créé. Pas de uv (choix utilisateur) ; Open WebUI écarté (choix utilisateur).

### Fait le 2026-10-09 (3e passe, choix utilisateur : tout sauf historique persistant)
- Pipeline factorisé (`rag_pipeline.retrieve_passages`) ; `rag_query` = version assemblée de `rag_query_stream` ; événements `status` dans le flux.
- Questions de suivi : `condense_question` (prompt `CONDENSE_PROMPT`, `HISTORY_TURNS`). Testé en réel : suivi OK, changement de sujet OK après ajustement du prompt.
- Seuil `RERANK_MIN_SCORE=0.01`, alerte `RELEVANCE_WARNING_SCORE=0.1` : calibrés sur un sondage (pertinent ~0.44-0.99, hors sujet 0). À affiner avec `evaluate.py`.
- Contexte élargi : `chunk_index` à l'indexation, voisins via `HybridRetriever.get_neighbors`, recouvrement fusionné (`retrieval.merge_overlapping`). Nécessite une réindexation (base actuellement vide de toute façon).
- Citations numérotées `[n]` ; interface : onglets par source, extrait, aperçu de page PDF (pypdfium2, cache), téléchargement différé (Streamlit >= 1.52), pastilles pertinence/OCR.
- Rerank : renvoie des copies (ne modifie plus les chunks partagés de l'index BM25) et trie lui-même par score.
- Formats : `.doc` (doc_reader.py, piece table Word 97, testé sur tests/fixtures/exemple.doc généré avec Word), `.eml` (corps + pièces jointes, signatures intégrées ignorées), `.md` lu comme texte (le paquet `markdown` manquait : aucun .md n'était indexé), ODT avec titres et sans pieds de page.
- Logging : `config.setup_logging()` ; plus de print dans les modules ; `except` nus remplacés.
- Indexation : fichiers dont un lot échoue retentés ; chunks résiduels des nouveaux fichiers purgés.
- `main.py` corrigé (affichait les dictionnaires du flux).
- Tests : `test_loaders.py` réécrit, `test_pipeline_qualite.py`, `test_retrieval.py`, `test_evaluate.py` ; erreurs de teardown Chroma corrigées (conftest). 74 passés.
- CI : `.github/workflows/tests.yml` (remote = GitHub) et `.gitlab-ci.yml`.

### Fait le 2026-10-09 (4e passe)
- `parametres.yaml` : source unique des réglages, lu et validé par `config.py` (mêmes constantes qu'avant ; `ParametresError` stable au rechargement).
- Avatar `assets/avatar.png` (script `assets/dessiner_avatar.py`) ; textes de l'interface dans `interface.*`.
- Attente : événements `status` (`step`, `detail`, `elapsed`) de `rag_pipeline.iter_retrieval_steps`, affichés par `st.status` + `st.spinner(show_time=True)`.
- Documents similaires (`corpus.py`) : métadonnées `meeting_date`, `validated`, `doc_title`, `doc_family` ; en-tête contextuel (`header_len`) ; `collapse_duplicates` (critère `near_duplicate` : Jaccard ou inclusion >= 0,85 si >= 80 empreintes) ; `cap_per_document` (famille et date, 2) ; filtres période (interrupteur « Filtrer par période », désactivé par défaut) et validées.
- Qualité des données : tableaux HTML de l'OCR en Markdown, boucles de répétition, consigne recopiée (`normalize_ocr_text`, appliqué à la lecture du cache) ; ODT sans commentaires ni texte supprimé.
- Évaluation réalisée : 55 questions écrites à partir des documents ; calibrage : HyDE désactivé (aucun gain, 14 s par question), score minimum 0,03, pastille orange 0,1 ; voir `evaluation/rapport_evaluation.md`. Recherche vectorielle seule faible (18 à 29 % de succès) : piste modèle d'embeddings.
- Notebook `exemple_pipeline.ipynb` généré par `_tmp/build_notebook.py --executer` (textes `_tmp/notebook_textes.md`) ; docs Quarto et README réécrits.

### Points ouverts
- Base de l'application `chroma_db_rag/` indexée AVANT ces changements (sans métadonnées de corpus, avec bruit OCR) : la remplacer par `_tmp/chroma_full` (application arrêtée) ou la vider en gardant `ocr_cache/` puis `python main.py`.
- `.coverage` et `rag_logs.db` sont suivis par git alors qu'ils sont dans `.gitignore`.
- Pistes : meilleur modèle d'embeddings ou passages plus courts pour la recherche vectorielle ; jeu de questions formulées par des agents (les questions actuelles reprennent le vocabulaire des documents, ce qui avantage BM25) ; évaluer la fidélité des réponses de façon outillée.

### Environnement
- `.venv` de test local ; installer avec `pip install -r requirements.txt -r requirements-dev.txt` (pas de uv).
- Tests : `.venv/Scripts/python -m pytest tests -q`.
