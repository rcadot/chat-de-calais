# 🐈 Chat de Calais

> Système de Retrieval-Augmented Generation (RAG) avancé avec HyDE et Reranking ALBERT

Un système RAG intelligent permettant d'interroger une base documentaire à l'aide de l'intelligence artificielle, optimisé pour les administrations publiques françaises.

## ✨ Fonctionnalités

### 🤖 Pipeline RAG Avancé
- **HyDE (Hypothetical Document Embeddings)** : Amélioration du retrieval via génération de documents hypothétiques
- **Retrieval Vectoriel** : Recherche sémantique dans ChromaDB
- **Reranking ALBERT** : Affinage de la pertinence avec l'API Etalab
- **Génération LLM** : Réponses contextualisées avec streaming

### 🎭 3 Modes de Prompt
- **Administratif** : Ton formel et réglementaire pour les contextes officiels
- **Technique** : Réponses détaillées avec procédures et exemples
- **Créatif** : Vulgarisation pédagogique avec analogies

### 📚 Gestion Documentaire
- **Indexation incrémentale** : Détection automatique des changements (hash MD5)
- **Multi-format** : PDF, DOCX, ODT, TXT, MD, HTML
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
│  1. HyDE        │  Génération document hypothétique
│  (LLM ALBERT)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  2. Retrieval   │  Top 30 documents similaires
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

Créer un fichier `.env` à la racine du projet :

```bash
# API ALBERT
ALBERT_API_KEY=votre_clé_api_ici

# Mode de prompt par défaut (administratif | technique | créatif)
PROMPT_MODE=administratif
```

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
- Sources consultées avec scores

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

## ⚙️ Configuration

Tous les paramètres sont centralisés dans `config.py` :

### API & Modèles

```python
EMBEDDINGS_MODEL = "embeddings-small"    # Modèle d'embeddings
LLM_MODEL = "albert-large"               # Modèle de génération
RERANK_MODEL = "rerank-small"            # Modèle de reranking
```

### Pipeline RAG

```python
USE_HYDE = True                          # Activer HyDE
USE_RERANK = True                        # Activer le reranking
RAG_TOP_K_DOCS = 5                       # Nombre de docs finaux
RAG_TOP_N_RETRIEVAL = 30                 # Nombre de docs récupérés
```

### Chunking

```python
CHUNK_SIZE = 2000                        # Taille des chunks
CHUNK_OVERLAP = 400                      # Chevauchement
CHUNK_SEPARATORS = ["\n\n", "\n", ". "]  # Séparateurs
```

### Prompts personnalisés

Modifier les templates dans `PROMPT_TEMPLATES` pour personnaliser les réponses par mode.

## 📁 Structure du projet

```
chat-de-calais/
├── 📄 config.py                  # Configuration centralisée
├── 🔌 albert_client.py           # Client API ALBERT
├── 📚 loaders.py                 # Chargeurs multi-formats
├── 🔍 indexer.py                 # Indexation incrémentale ChromaDB
├── 🤖 rag_pipeline.py            # Pipeline RAG (HyDE + Rerank)
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
├── 📖 Documentation
│   ├── README.md                 # Ce fichier
│   ├── docs.qmd                  # Documentation technique Quarto
│   ├── exemple_pipeline.ipynb    # Notebook de démonstration
│   └── requirements.txt          # Dépendances Python
│
└── 🗄️  Données (générées)
    ├── documents/                # Documents à indexer
    ├── chroma_db_rag/            # Base vectorielle
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

1. **HyDE** génère un document hypothétique sur les règles d'urbanisme
2. **Retrieval** récupère 30 documents pertinents via embeddings
3. **Reranking** sélectionne les 5 documents les plus pertinents (scores : 0.92, 0.89, 0.85, 0.82, 0.78)
4. **Génération** produit une réponse structurée avec sources

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

📚 Sources consultées:
[0.920] Guide_urbanisme_2024.pdf
[0.890] PLU_extensions_habitations.pdf
[0.850] Procedures_declaratives.pdf
```

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
- 📚 **Documentation** : Voir `docs.qmd` et `exemple_pipeline.ipynb`

---
