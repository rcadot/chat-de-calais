# Dictionnaire de données

## Chunk indexé (collection Chroma `documents_rag`, métadonnées)

| Champ | Type | Origine | Description |
|---|---|---|---|
| `source` | str | loaders | Chemin du fichier d'origine (clé de suppression lors de la réindexation) ; pour une pièce jointe, chemin du `.eml` |
| `attachment` | str | `loaders.load_eml` | Nom de la pièce jointe d'un courriel ; absent sinon |
| `chunk_index` | int | `indexer.split_documents` | Rang du chunk dans son fichier (ordre de lecture), sert au contexte élargi |
| `meeting_date` | int | `corpus.meeting_date` | Date de réunion AAAAMMJJ tirée du dossier « jj mm aaaa » le plus proche ; 0 si inconnue. Filtre par période (`$gte`, `$lte`) |
| `validated` | bool | `corpus.is_validated` | Vrai si le chemin contient validée, finale, définitive ou signée (`documents.mots_version_validee`) |
| `doc_title` | str | `corpus.document_title` | Nom du fichier sans numéro d'ordre ni soulignés ; utilisé dans l'en-tête contextuel |
| `doc_family` | str | `corpus.document_family` | Clé commune aux versions et formats d'une note (sans accents, numéros d'ordre, dates ni marqueurs de version) ; plafond de passages par document avec `meeting_date` |
| `header_len` | int | `indexer.split_documents` | Longueur de l'en-tête contextuel placé devant le texte (`[titre \| réunion du jj/mm/aaaa \| version validée]`) ; `corpus.chunk_body` le retire |
| `page` | int | PyPDF, OCR | Indice de page, base 0 (affiché en base 1 dans le contexte) |
| `ocr` | bool | `loaders.load_pdf`, `load_image` | Vrai si le texte provient de l'OCR |
| `ocr_images` | int | `loaders.ocr_image_regions` | Nombre d'images d'une page de texte dont le texte a été ajouté (PDF mixtes) |
| `ocr_model` | str | idem | Modèle OCR utilisé (`openweight-ocr` ou `mistral-ocr-2512`) ; absent sans OCR |
| `rerank_score` | float | `rag_pipeline.rerank_documents` | Score de pertinence ajouté à la volée sur une copie, non persisté dans Chroma |
| `aussi_dans` | str | `corpus.collapse_duplicates` | À la recherche seulement : autres fichiers contenant un passage quasi identique (« fichier (jj/mm/aaaa) ; ... ») |
| autres (`producer`, `creator`, `total_pages`...) | divers | PyPDF | Métadonnées PDF brutes |

## `chroma_db_rag/index_metadata.json`

| Champ | Type | Description |
|---|---|---|
| `files.<chemin>.size` | int | Taille en octets |
| `files.<chemin>.mtime` | float | Date de modification (epoch) |
| `files.<chemin>.hash` | str | MD5 du contenu, sert à détecter les modifications |
| `last_update` | str | Horodatage ISO de la dernière indexation |

Les fichiers n'ayant produit aucun chunk, ou dont l'ajout a échoué, ne figurent pas dans `files` (retentés au lancement suivant). Les fichiers exclus par `documents.exclure` ne sont pas scannés.

## `chroma_db_rag/ocr_cache/<md5>_<moteur>.json`

Dictionnaire de textes bruts renvoyés par l'OCR. Clés : `"<page>"` pour une page entière, `"<page>_img<k>"` pour la k-ième image retenue d'une page de texte. Les éléments en échec n'y sont pas inscrits et sont donc retentés. Le nettoyage (`loaders.normalize_ocr_text` : tableaux HTML en Markdown, boucles de répétition, consigne recopiée) s'applique à la lecture : le cache garde le texte d'origine.

## Index BM25 (`retrieval.BM25Index`, en mémoire)

Reconstruit au démarrage depuis la collection Chroma (ou depuis les documents temporaires). Termes : minuscules, sans accents, sans mots vides, nombres conservés (`retrieval.tokenize_fr`).

## Passage cité (`rag_pipeline.build_passages`, stocké dans `st.session_state.messages[i]["passages"]`)

| Clé | Type | Description |
|---|---|---|
| `num` | int | Numéro de citation, identique au `[n]` du contexte et de la réponse |
| `source` | str | Chemin du fichier |
| `label` | str | Libellé affiché : `fichier, p. N` ou `courriel.eml > pièce.pdf` |
| `details` | str | Date de réunion et statut : `réunion du jj/mm/aaaa \| version validée` |
| `aussi_dans` | str or None | Autres fichiers contenant le même passage |
| `page` | int or None | Page, base 0 |
| `score` | float or None | Score de rerank |
| `excerpt` | str | Texte du chunk retenu, sans en-tête contextuel ni voisins, tableaux OCR en Markdown |
| `ocr`, `ocr_images`, `attachment` | bool, int, str | Repris des métadonnées du chunk |

## Métadonnées renvoyées par `rag_query` et `rag_query_stream`

| Clé | Type | Description |
|---|---|---|
| `sources` | list[str] | `source` des documents finaux |
| `rerank_scores` | list[float or None] | Scores de rerank correspondants |
| `n_docs_retrieved` | int | Nombre de documents avant dédoublonnage et rerank |
| `n_docs_final` | int | Nombre de documents transmis au LLM |
| `retrieval_time` | float or None | Durée de la recherche (s), avant la rédaction |
| `execution_time` | float | Durée totale (s) |
| `prompt_mode` | str | `administratif`, `technique` ou `créatif` |
| `error` | str or None | Message d'erreur éventuel |
| `standalone_query` | str | Question reformulée à partir de l'historique (identique à la question sans historique) |
| `passages` | list[dict] | Passages cités (voir ci-dessus) |
| `no_relevant_docs` | bool | Vrai si aucun passage n'a passé `RERANK_MIN_SCORE` (réponse non générée) |
| `hyde_query` | str | Requête enrichie par HyDE |
| `retrieved_docs`, `reranked_docs` | list[Document] | Documents avant et après rerank |

## Événements de statut (`rag_query_stream`, `iter_retrieval_steps`)

| Clé | Type | Description |
|---|---|---|
| `type` | str | `status` |
| `step` | str | `reformulation`, `recherche`, `tri`, `redaction` (libellés dans `interface.etapes`) |
| `content` | str | Libellé de l'étape |
| `detail` | str (facultatif) | Détail chiffré (« 30 passages candidats dans 12 document(s) »...) |
| `elapsed` | float | Secondes écoulées depuis le début de la question |

## Filtres de recherche (`filters`)

| Clé | Type | Description |
|---|---|---|
| `date_min`, `date_max` | int | Bornes AAAAMMJJ sur `meeting_date` |
| `validated_only` | bool | Versions validées uniquement |

## Résultats d'évaluation (`evaluation/resultats/*.csv`, séparateur `;`)

| Colonne | Description |
|---|---|
| `question`, `sources_attendues` | Question et fragments de noms de fichiers attendus (`(aucune)` : question hors sujet) |
| `succes` | Source attendue parmi les passages retenus, ou aucun passage pour une question hors sujet |
| `rang_recherche`, `rang_final` | Rang de la première source attendue avant et après rerank |
| `n_passages`, `meilleur_score` | Nombre de passages retenus et meilleur score de rerank |
| `documents_distincts`, `doublons` | Diversité des passages retenus et nombre de passages quasi identiques |
| `duree_s` | Durée de la recherche (s) |
| `sources_retenues` | Fichiers des passages retenus |
| `succes_seuil_<x>` | Succès simulé avec le seuil de pertinence x (option `--seuils`) |
| `reponse` | Réponse générée (option `--generate`) |
