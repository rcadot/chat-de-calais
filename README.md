# 🐈 Chat de Calais

> Système de Retrieval-Augmented Generation (RAG) avancé avec recherche hybride et Reranking ALBERT

Un système RAG intelligent permettant d'interroger une base documentaire à l'aide de l'intelligence artificielle, optimisé pour les administrations publiques françaises.

## 🆕 Nouveautés de la version d'octobre 2026

### Fond : qualité des réponses
- Recherche hybride : sémantique (ChromaDB) et par mots-clés (BM25), fusionnées par RRF (`retrieval.py`). Elle retrouve les références exactes (articles, Cerfa, noms de communes).
- Questions de suivi reformulées en questions autonomes à partir de l'historique.
- Seuil de pertinence après le reranking Albert : sans passage pertinent, l'assistant l'indique et n'appelle pas le modèle.
- Contexte élargi : chaque passage retenu est complété par ses voisins dans le document.
- Citations numérotées `[1]`, `[2]` dans la réponse, en correspondance avec les sources affichées.
- Documents similaires (`corpus.py`) : date de réunion et statut « validé » déduits des noms de dossiers et de fichiers, en-tête contextuel devant chaque passage, regroupement des passages quasi identiques (version validée la plus récente conservée, autres versions signalées « aussi dans »), au plus 2 passages par document.
- HyDE désactivé par défaut (aucun gain mesuré, environ 15 s d'attente en plus par question).
- Évaluation sur 55 questions (49 documentées, 6 hors sujet) : 49 sur 49 retrouvées, 6 sur 6 rejetées, MRR 0,97, 0,4 s par recherche. Sans recherche hybride, 18 % de succès. Sans rerank, aucune question hors sujet rejetée. Limite : questions écrites avec le vocabulaire des documents, ce qui avantage la recherche par mots-clés. Voir `evaluation/rapport_evaluation.md` et `evaluate.py`.

### Documents et données
- OCR par Albert (`openweight-ocr`, ou `mistral-ocr-2512` avec repli) des PDF scannés, des images insérées dans les pages de texte (10 à 85 % de la page) et des fichiers PNG, JPG, TIFF. Cache disque (`ocr_cache/`), nettoyage du texte (tableaux en Markdown, boucles de répétition supprimées).
- Nouveaux formats : `.doc` (Word 97-2003, lu sans LibreOffice par `doc_reader.py`), `.eml` (corps et pièces jointes), `.md` (non indexé auparavant faute d'un paquet manquant). ODT lus sans pieds de page, commentaires ni texte supprimé du suivi des modifications.
- Indexation plus robuste : fichier en échec retenté au lancement suivant, reliquats purgés, exclusion possible via `documents.exclure` (ex. ordres du jour).

### Interface
- Logo en haut de la barre latérale et en icône d'onglet (chat et tableau de bord des logs).
- Statut animé pendant le traitement : étape en cours, détail, temps écoulé.
- Sources en onglets numérotés comme les citations : extrait, date de réunion, statut de version, aperçu de la page PDF, téléchargement, pastilles de pertinence et d'OCR, mention « aussi dans ».
- Filtres : « Versions validées uniquement » et « Filtrer par période » (désactivé par défaut pour ne pas écarter les documents sans date ou indexés récemment).
- Documents déposés par glisser-déposer : valables pour la session seulement (base en mémoire), interrogés avec la base permanente ou seuls (« Interroger uniquement ces documents »).
- Textes d'accueil et d'interface modifiables sans code.

### Paramétrage
- `parametres.yaml` : fichier unique commenté (sections `albert`, `documents`, `base`, `decoupage`, `recherche`, `ocr`, `interface`, `prompts`), validé au démarrage avec un message qui désigne la clé fautive.
- `.env` réservé à `ALBERT_API_KEY` et aux surcharges ponctuelles (une variable d'environnement prime sur le YAML). `PROMPT_MODE` y devient inutile (`interface.mode_par_defaut`). Modèle fourni dans `.env.example`.

### Code
- Nouveaux modules : `corpus.py`, `retrieval.py`, `doc_reader.py`, `evaluate.py`.
- Pipeline factorisé en générateur d'étapes (`rag_pipeline.iter_retrieval_steps`), réutilisé par l'interface, l'évaluation et le notebook. Rerank sans effet de bord, journalisation par `logging` au lieu de `print`, `main.py` corrigé (affichait des dictionnaires bruts).
- Dépendances mises à jour dans `requirements.txt` et `requirements-dev.txt` (pip, pas de uv). Streamlit 1.52 au minimum.

### Tests et intégration continue
- 102 tests (1 ignoré), sans appel à Albert (API simulée). Nouveaux fichiers pour le corpus, l'OCR, les paramètres, la recherche, l'évaluation et la qualité du pipeline.
- Intégration continue : GitHub Actions (`.github/workflows/tests.yml`) et GitLab CI (`.gitlab-ci.yml`).

### Documentation
- `README.md`, documentation technique Quarto (`docs/docs.qmd`, `docs/docs.html`), présentation (`docs/presentation.qmd`), notebook `exemple_pipeline.ipynb` exécuté, `dictionnaire_donnees.md`, `evaluation/rapport_evaluation.md`, notes de reprise `memory.md`.
- Dépôt : `rag_logs.db` (questions réelles) et `.coverage` ne sont plus suivis par git, l'ancienne base vectorielle est ignorée.

### Action requise après mise à jour
- La base `chroma_db_rag/` a été indexée avant ces changements. Pour bénéficier des métadonnées de réunion, du regroupement des documents similaires et du nettoyage OCR, la réindexer complètement : vider le dossier en conservant `chroma_db_rag/ocr_cache/`, puis lancer `python main.py` (une vingtaine de secondes grâce au cache OCR).
- Pistes ouvertes : meilleur modèle d'embeddings (la recherche sémantique seule n'atteint que 18 à 29 % de succès), questions de test formulées autrement, évaluation outillée de la fidélité des réponses.

## ✨ Fonctionnalités

### 🤖 Pipeline RAG Avancé
- **Questions de suivi** : « et pour Calais ? » est reformulée en question autonome à partir de l'historique
- **HyDE (Hypothetical Document Embeddings)** : option, désactivée par défaut, qui recherche à partir d'un document hypothétique ; aucun gain mesuré avec la recherche hybride et environ 15 s d'attente en plus par question
- **Recherche hybride** : sémantique (ChromaDB) et par mots-clés (BM25), fusionnées par RRF ; retrouve les références exactes (articles, Cerfa, communes)
- **Reranking ALBERT** : Affinage de la pertinence avec l'API Etalab
- **Seuil de pertinence** : sans passage pertinent, l'assistant le dit au lieu d'inventer
- **Contexte élargi** : chaque passage est complété par ses voisins immédiats dans le document
- **Génération LLM** : Réponses en streaming, citations numérotées [1], [2]...

### 🎭 3 Modes de Prompt
- **Administratif** : Ton formel et réglementaire pour les contextes officiels
- **Technique** : Réponses détaillées avec procédures et exemples
- **Créatif** : Vulgarisation pédagogique avec analogies

### 📚 Gestion Documentaire
- **Indexation incrémentale** : Détection automatique des changements (hash MD5)
- **Multi-format** : PDF, DOCX, DOC (Word 97-2003, sans LibreOffice), ODT, TXT, MD, HTML, EML (courriels et pièces jointes), images (PNG, JPG, TIFF)
- **OCR des PDF scannés** : pages sans couche texte transcrites par Albert (`openweight-ocr`, ou `mistral-ocr-2512` si `OCR_ENGINE=mistral`), avec cache disque
- **PDF mixtes** : le texte des images insérées dans les pages (schémas, tableaux scannés) est aussi lu par OCR
- **Nettoyage de l'OCR** : tableaux HTML convertis en Markdown, boucles de répétition supprimées
- **Documents similaires** : la date de réunion et le statut de version validée sont tirés des noms de dossiers et de fichiers ; un en-tête contextuel est placé devant chaque passage ; les passages quasi identiques (versions successives, PDF et ODT d'une même note) sont regroupés en gardant la version validée la plus récente, les autres étant signalés par la mention « aussi dans » ; au plus 2 passages par document sont retenus
- **Documents temporaires** : Upload à la volée pour une session
- **Chunking intelligent** : Découpage optimisé avec chevauchement

### 💬 Feedback & Analytics
- **Feedback utilisateur** : Système de thumbs up/down
- **Dashboard interactif** : Visualisations avec Plotly
- **Logging complet** : Traçabilité de toutes les requêtes
- **Métriques de performance** : Temps, scores, satisfaction

## 🏗️ Architecture

```
┌─────────────┐
│   Question  │
└──────┬──────┘
       │
       v
┌─────────────────┐
│  1. HyDE        │  Facultatif (désactivé par défaut)
│  (LLM ALBERT)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  2. Retrieval   │  30 par méthode (vectoriel + BM25)
│  (ChromaDB)     │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  3. Reranking   │  Affinage → Top 5 documents
│  (ALBERT API)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  4. Génération  │  Réponse finale avec contexte
│  (LLM ALBERT)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│    Réponse      │ + Sources + Scores
└─────────────────┘
```

## 🚀 Installation

### Prérequis

- Python 3.10+
- Clé API ALBERT (https://albert.api.etalab.gouv.fr/)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://gitlab.cerema.fr/romain.cadot/chat-de-calais.git
# ou 
git clone https://github.com/rcadot/chat-de-calais.git

cd chat-de-calais

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

Copier `.env.example` en `.env` puis renseigner la clé :

```bash
cp .env.example .env
```

Les autres variables (OCR, recherche hybride, prompts personnalisés) sont documentées dans `.env.example`. Les réglages courants (modèles, recherche, OCR, textes de l'interface, prompts) se trouvent dans `parametres.yaml` (voir la section Paramétrage).

## 📖 Utilisation

### 1. Indexer les documents

Placer vos documents dans le dossier `./documents/` puis lancer :

```bash
python main.py
```

L'indexation est **incrémentale** : seuls les fichiers nouveaux ou modifiés seront traités.

### 2. Lancer l'application de chat

```bash
streamlit run app_chat.py
```

Accéder à l'interface : http://localhost:8501

**Fonctionnalités de l'interface :**
- Chat interactif avec historique
- Sélection du mode de prompt
- Upload de documents temporaires
- Feedback sur les réponses (👍/👎)
- Statut animé pendant le traitement : étapes, détails et temps écoulé
- Filtres de la barre latérale : période des réunions, versions validées uniquement
- Avatar et textes d'accueil définis dans `parametres.yaml`
- Sources numérotées comme les citations : extrait utilisé, date de réunion et statut de la version, aperçu de la page PDF, téléchargement du document, pastilles de pertinence et d'OCR

### 3. Lancer le dashboard analytics

```bash
streamlit run app_logs.py
```

Accéder au dashboard : http://localhost:8502

**Métriques disponibles :**
- Nombre de requêtes et taux de succès
- Temps d'exécution moyen
- Distribution des modes de prompt
- Feedbacks utilisateurs et satisfaction
- Top sources consultées
- Évolution temporelle

### 4. Consulter les logs (CLI)

```bash
# Voir les dernières requêtes
python view_logs.py recent --limit 10

# Détail d'une requête spécifique
python view_logs.py detail 42

# Statistiques globales
python view_logs.py stats

# Recherche dans les logs
python view_logs.py search "urbanisme" --limit 5
```

## ⚙️ Paramétrage

Tout se règle dans `parametres.yaml`, fichier commenté à l'intention d'un lecteur non développeur. Il comporte huit sections : `albert` (adresse et modèles), `documents`, `base`, `decoupage`, `recherche`, `ocr`, `interface` et `prompts`.

Après une modification, il suffit de relancer l'application. Les réglages marqués `[RÉINDEXATION]` ne s'appliquent qu'aux documents indexés ensuite : pour les appliquer à tout le corpus, vider le dossier de la base. Une erreur de saisie (clé mal orthographiée, mauvais type, variable de prompt manquante) interrompt le démarrage avec un message qui désigne la clé fautive, ou la ligne en cas d'erreur de syntaxe YAML.

La clé d'API et les surcharges ponctuelles (par exemple `OCR_ENGINE` ou `USE_RERANK`) se placent dans `.env` (voir `.env.example`) ; une variable d'environnement l'emporte sur `parametres.yaml`.

Quelques modifications courantes :

```yaml
# Changer de modèle de réponse
albert:
  modele_reponse: openweight-medium
```

```yaml
# Modifier le nombre de passages retenus
recherche:
  passages_retenus: 5
```

```yaml
# Exclure les ordres du jour de l'indexation
documents:
  exclure: ["ODJ*"]
```

Dans un prompt, les variables `{context}` et `{query}` doivent être conservées (`{history}` pour la reformulation) :

```yaml
prompts:
  modes:
    administratif:
      reponse: |
        Contexte : {context}
        Question : {query}
        Réponds en citant les extraits par leur numéro.
```

## 📁 Structure du projet

```
chat-de-calais/
├── 📄 config.py                  # Lecture et validation des paramètres
├── ⚙️  parametres.yaml            # Réglages modifiables sans code
├── 🔌 albert_client.py           # Client API ALBERT
├── 📚 loaders.py                 # Chargeurs multi-formats et OCR
├── 📄 doc_reader.py              # Lecture des .doc Word 97-2003
├── 🗂️  corpus.py                  # Règles pour les documents similaires (métadonnées, doublons, filtres)
├── 🔍 indexer.py                 # Indexation incrémentale ChromaDB
├── 🔎 retrieval.py               # Recherche hybride (BM25 + vectoriel)
├── 🤖 rag_pipeline.py            # Pipeline RAG (reformulation, HyDE, rerank, génération)
├── 📏 evaluate.py                # Évaluation sur questions de référence
├── 📝 logger.py                  # Logging SQLite
│
├── 🖥️  Applications Streamlit
│   ├── app_chat.py               # Interface de chat
│   ├── app_logs.py               # Dashboard analytics
│   ├── temp_documents.py         # Gestion docs temporaires
│   └── utils_app.py              # Utilitaires UI
│
├── 🛠️  Scripts utilitaires
│   ├── main.py                   # Script d'indexation
│   ├── view_logs.py              # CLI de consultation logs
│   └── generate_mock_logs.py     # Génération de logs de test
│
├── 🖼️  assets/                    # Avatar de l'interface
├── 📏 evaluation/                # Questions de référence, résultats et rapport
├── 📖 Documentation
│   ├── README.md                 # Ce fichier
│   ├── dictionnaire_donnees.md   # Dictionnaire de données
│   ├── exemple_pipeline.ipynb    # Notebook pédagogique
│   ├── requirements.txt          # Dépendances Python
│   └── docs/                     # docs.qmd (documentation technique), presentation.qmd
│
└── 🗄️  Données (générées)
    ├── documents/                # Documents à indexer
    ├── chroma_db_rag/            # Base vectorielle et cache OCR
    └── rag_logs.db               # Base de logs SQLite
```

## 🔧 Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Framework RAG** | LangChain |
| **Base vectorielle** | ChromaDB |
| **LLM & Embeddings** | ALBERT (API Etalab) |
| **Interface Web** | Streamlit |
| **Base de données** | SQLite |
| **Visualisations** | Plotly |
| **Chargeurs documents** | PyPDF, Docx2txt, Unstructured |

## 📊 Exemple de requête

### Question
> "Quelles sont les règles d'urbanisme pour construire une extension de maison ?"

### Pipeline (mode technique)

1. **HyDE** (facultatif, désactivé par défaut) génèrerait un document hypothétique sur les règles d'urbanisme ; sans lui, la recherche utilise directement la question
2. **Retrieval** récupère 30 passages par méthode (embeddings et BM25), fusionnés par RRF
3. **Reranking** sélectionne les 5 passages les plus pertinents, ceux dont le score est sous le seuil étant écartés
4. **Génération** produit une réponse structurée, dont les sources sont citées par numéro [1], [2]

### Résultat

```
Pour construire une extension de maison, vous devez respecter plusieurs règles :

1. Déclaration préalable de travaux
   - Si surface < 20m² (40m² en zone urbaine PLU)
   - Formulaire Cerfa 13703

2. Permis de construire
   - Si surface > 20m² (40m² en zone urbaine)
   - Délai d'instruction : 2 mois

3. Règles d'urbanisme locales
   - Consulter le PLU de votre commune
   - Respect des distances par rapport aux limites
   - Hauteur maximale autorisée

Sources : [1] Guide_urbanisme_2024.pdf, [2] PLU_extensions_habitations.pdf, [3] Procedures_declaratives.pdf
```

## 📏 Évaluation de la qualité

Le fichier `evaluation/questions.yaml` contient 55 questions rédigées à partir des documents (49 documentées, 6 hors sujet). Les résultats figurent dans `evaluation/rapport_evaluation.md`. Sur la configuration par défaut, les 49 questions documentées sont toutes retrouvées, les 6 questions hors sujet sont rejetées et le MRR vaut 0,97, pour 0,4 s par recherche. La recherche hybride (18 % de succès sans elle) et le rerank (aucun rejet des questions hors sujet sans lui) sont indispensables, tandis que HyDE, sans gain mesuré, a été désactivé. Ces résultats reposent sur des questions rédigées avec le vocabulaire des documents. Pour relancer l'évaluation :

```bash
python evaluate.py                       # réglages courants
python evaluate.py --no-hybrid           # comparer sans recherche hybride
python evaluate.py --no-dedup            # comparer sans regroupement des documents similaires
python evaluate.py --min-score 0.05      # tester un autre seuil de pertinence
python evaluate.py --seuils 0.005,0.03   # simuler plusieurs seuils sans nouvel appel
python evaluate.py --generate            # produire aussi les réponses
```

Le script affiche le taux de succès, le rappel de la recherche, le MRR et le taux de questions hors sujet correctement rejetées ; le détail est écrit dans `evaluation/resultats/`.

## 🧪 Tests

Lancer les tests unitaires :

```bash
pytest tests/ -v
```



## 👥 Auteurs

- **Romain Cadot** - Développement initial

## 🙏 Remerciements

- [ALBERT](https://albert.etalab.gouv.fr/) - API d'IA générative pour l'administration publique
- [LangChain](https://www.langchain.com/) - Framework pour applications LLM
- [Streamlit](https://streamlit.io/) - Framework d'applications web Python

## 📞 Support

Pour toute question ou problème :

- 🐛 **Issues** : [GitLab Issues](https://gitlab.cerema.fr/romain.cadot/chat-de-calais/-/issues)
- 📧 **Email** : romain.cadot@cerema.fr
- 📚 **Documentation** : documentation technique `docs/docs.html` (source `docs/docs.qmd`), présentation `docs/presentation.qmd` et notebook `exemple_pipeline.ipynb`

---
