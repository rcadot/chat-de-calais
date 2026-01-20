# generate_mock_logs.py
"""Génère des logs mock pour tester le dashboard."""

import sqlite3
import random
from datetime import datetime, timedelta
import json
from pathlib import Path
from logger import RAGLogger

# Configuration
logger = RAGLogger()
DB_PATH = logger.db_path
NUM_QUERIES = 100  # Nombre de requêtes à générer

# Données mock
QUESTIONS = [
    "Comment déposer une demande de permis de construire ?",
    "Quels sont les horaires de la mairie ?",
    "Comment obtenir un acte de naissance ?",
    "Où se trouve la déchetterie la plus proche ?",
    "Comment s'inscrire sur les listes électorales ?",
    "Quelles sont les démarches pour une carte d'identité ?",
    "Comment payer mes impôts locaux ?",
    "Où trouver les informations sur le PLU ?",
    "Comment signaler un problème de voirie ?",
    "Quels sont les services disponibles en ligne ?",
    "Comment obtenir une autorisation de stationnement ?",
    "Où se trouve la bibliothèque municipale ?",
    "Comment inscrire mon enfant à l'école ?",
    "Quelles sont les aides sociales disponibles ?",
    "Comment déclarer un changement d'adresse ?",
    "Où se trouve le centre social ?",
    "Comment obtenir un extrait de casier judiciaire ?",
    "Quels sont les marchés de la ville ?",
    "Comment participer au conseil municipal ?",
    "Où trouver les informations sur les transports ?",
]

ANSWERS = [
    "Pour déposer une demande de permis de construire, vous devez vous rendre en mairie avec les documents nécessaires. Le dossier doit contenir les plans du projet et une description détaillée.",
    "La mairie est ouverte du lundi au vendredi de 8h30 à 12h00 et de 13h30 à 17h00. Le samedi, l'accueil est ouvert de 9h00 à 12h00.",
    "Pour obtenir un acte de naissance, vous pouvez faire une demande en ligne sur le site de la mairie ou vous présenter directement au service état civil avec une pièce d'identité.",
    "La déchetterie municipale se trouve Zone Industrielle Nord, rue des Artisans. Elle est ouverte du mardi au samedi de 9h à 18h.",
    "Pour vous inscrire sur les listes électorales, rendez-vous en mairie avec une pièce d'identité et un justificatif de domicile de moins de 3 mois.",
]

SOURCES = [
    "reglement_urbanisme.pdf",
    "guide_demarches_administratives.pdf",
    "horaires_services_municipaux.pdf",
    "plan_local_urbanisme.pdf",
    "deliberations_conseil_municipal.pdf",
    "reglement_interieur.pdf",
    "guide_citoyen.pdf",
    "annuaire_services.pdf",
    "charte_accueil.pdf",
    "statuts_associations.pdf",
]

MODES = ["administratif", "technique", "créatif"]

HYDE_QUERIES = [
    "Procédure administrative pour déposer une demande de permis de construire auprès des services municipaux",
    "Horaires d'ouverture et plages de disponibilité des services de la mairie pour le public",
    "Démarches et documents nécessaires pour l'obtention d'un acte de naissance officiel",
]


def create_mock_database():
    """Crée et remplit la base de données avec des données mock."""

    # Supprimer la DB existante si elle existe
    if Path(DB_PATH).exists():
        response = input(f"{DB_PATH} existe déjà. Voulez-vous la supprimer ? (o/n): ")
        if response.lower() == "o":
            Path(DB_PATH).unlink()
            print(f"{DB_PATH} supprimée")
        else:
            print("Opération annulée")
            return

    # Créer la connexion
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Créer la table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_query TEXT NOT NULL,
            hyde_query TEXT,
            retrieved_docs_count INTEGER,
            reranked_docs_count INTEGER,
            final_answer TEXT,
            sources TEXT,
            rerank_scores TEXT,
            execution_time_seconds REAL,
            error TEXT,
            retrieved_docs_details TEXT,
            reranked_docs_details TEXT,
            prompt_mode TEXT,
            user_feedback TEXT,
            feedback_timestamp TEXT
        )
        """
    )

    print(f" Génération de {NUM_QUERIES} requêtes mock...")

    # Générer les requêtes
    now = datetime.now()

    for i in range(NUM_QUERIES):
        # Timestamp aléatoire dans les 30 derniers jours
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)

        # Question et réponse
        question = random.choice(QUESTIONS)
        answer = random.choice(ANSWERS)
        hyde_query = random.choice(HYDE_QUERIES)

        # Mode
        mode = random.choice(MODES)

        # Nombre de docs
        retrieved_count = random.randint(8, 15)
        reranked_count = random.randint(3, 6)

        # Sources
        num_sources = random.randint(2, 5)
        selected_sources = random.sample(SOURCES, num_sources)
        sources_json = json.dumps(selected_sources)

        # Scores de rerank
        rerank_scores = [
            round(random.uniform(0.3, 0.95), 4) for _ in range(num_sources)
        ]
        rerank_scores.sort(reverse=True)
        scores_json = json.dumps(rerank_scores)

        # Temps d'exécution
        execution_time = round(random.uniform(1.5, 8.0), 2)

        # Erreur (10% de chance)
        error = None
        if random.random() < 0.1:
            error = random.choice(
                [
                    "Timeout lors de la récupération des documents",
                    "Erreur de connexion au LLM",
                    "Score de rerank trop faible",
                ]
            )
            answer = "Désolé, une erreur s'est produite."

        # Feedback (60% de chance d'avoir un feedback)
        user_feedback = None
        feedback_timestamp = None
        if random.random() < 0.6:
            user_feedback = random.choice(["thumbs_up", "thumbs_down"])
            # Feedback donné quelques minutes après la requête
            feedback_delay = timedelta(minutes=random.randint(1, 30))
            feedback_timestamp = (timestamp + feedback_delay).isoformat()

            # Ajuster les probabilités : plus de thumbs_up pour les réponses sans erreur
            if error is None and random.random() < 0.75:
                user_feedback = "thumbs_up"

        # Détails des documents (simplifiés)
        retrieved_details = json.dumps(
            [
                {"source": s, "content": f"Extrait du document {s[:20]}..."}
                for s in random.sample(SOURCES, min(3, retrieved_count))
            ]
        )

        reranked_details = json.dumps(
            [
                {
                    "source": s,
                    "content": f"Extrait du document {s[:20]}...",
                    "rerank_score": score,
                }
                for s, score in zip(selected_sources, rerank_scores)
            ]
        )

        # Insérer dans la base
        cursor.execute(
            """
            INSERT INTO rag_queries (
                timestamp, user_query, hyde_query, retrieved_docs_count,
                reranked_docs_count, final_answer, sources, rerank_scores,
                execution_time_seconds, error, retrieved_docs_details,
                reranked_docs_details, prompt_mode, user_feedback, feedback_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp.isoformat(),
                question,
                hyde_query,
                retrieved_count,
                reranked_count,
                answer,
                sources_json,
                scores_json,
                execution_time,
                error,
                retrieved_details,
                reranked_details,
                mode,
                user_feedback,
                feedback_timestamp,
            ),
        )

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{NUM_QUERIES} requêtes générées")

    conn.commit()
    conn.close()

    print(f"\nBase de données mock créée : {DB_PATH}")
    print(f" {NUM_QUERIES} requêtes générées")
    print(f" Distribution des modes : {', '.join(MODES)}")
    print(f" ~{int(NUM_QUERIES * 0.6)} feedbacks utilisateurs")
    print(f" ~{int(NUM_QUERIES * 0.1)} erreurs simulées")
    print(f"\n Lancez 'streamlit run app_logs.py' pour voir le dashboard !")


if __name__ == "__main__":
    create_mock_database()
