# app_logs.py (VERSION AVEC FEEDBACK UTILISATEUR)
"""Dashboard Streamlit pour visualiser les logs RAG avec modes de prompt et feedback."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
from logger import RAGLogger

st.set_page_config(page_title="Dashboard RAG Logs", page_icon="📊", layout="wide")

# Init logger
logger = RAGLogger()

st.title("📊 Dashboard des Logs RAG")

# Sidebar
with st.sidebar:
    st.header("🔍 Filtres")

    # Date range
    period = st.selectbox(
        "Période",
        [
            "Dernière heure",
            "Aujourd'hui",
            "7 derniers jours",
            "30 derniers jours",
            "Tout",
        ],
    )

    # Status filter
    status_filter = st.radio(
        "Status", ["Tous", "Succès uniquement", "Erreurs uniquement"]
    )

    # Mode filter
    mode_filter = st.multiselect(
        "Modes de prompt",
        ["administratif", "technique", "créatif"],
        default=["administratif", "technique", "créatif"],
    )

    # ✅ NOUVEAU: Feedback filter
    feedback_filter = st.radio(
        "Feedback utilisateur",
        ["Tous", "👍 Positif uniquement", "👎 Négatif uniquement", "Sans feedback"],
    )

    # Refresh button
    if st.button("🔄 Rafraîchir"):
        st.rerun()


# Charger les données
@st.cache_data(ttl=60)
def load_data():
    import sqlite3

    conn = sqlite3.connect(logger.db_path)
    df = pd.read_sql_query(
        """
        SELECT 
            id, timestamp, user_query, retrieved_docs_count, 
            reranked_docs_count, execution_time_seconds, error, 
            sources, rerank_scores, prompt_mode, 
            user_feedback, feedback_timestamp, final_answer  
        FROM rag_queries 
        ORDER BY timestamp DESC
        """,
        conn,
    )
    conn.close()

    # Convertir timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["has_error"] = df["error"].notna()

    # Parser les scores de rerank
    def parse_scores(scores_json):
        if pd.isna(scores_json):
            return None
        try:
            scores = json.loads(scores_json)
            valid_scores = [s for s in scores if s is not None]
            return valid_scores if valid_scores else None
        except:
            return None

    df["rerank_scores_parsed"] = df["rerank_scores"].apply(parse_scores)
    df["avg_rerank_score"] = df["rerank_scores_parsed"].apply(
        lambda x: sum(x) / len(x) if x and len(x) > 0 else None
    )
    df["min_rerank_score"] = df["rerank_scores_parsed"].apply(
        lambda x: min(x) if x and len(x) > 0 else None
    )

    return df


df = load_data()

# Filtrer par période
if period != "Tout":
    now = datetime.now()
    if period == "Dernière heure":
        df = df[df["timestamp"] >= now - timedelta(hours=1)]
    elif period == "Aujourd'hui":
        df = df[df["timestamp"].dt.date == now.date()]
    elif period == "7 derniers jours":
        df = df[df["timestamp"] >= now - timedelta(days=7)]
    elif period == "30 derniers jours":
        df = df[df["timestamp"] >= now - timedelta(days=30)]

# Filtrer par status
if status_filter == "Succès uniquement":
    df = df[~df["has_error"]]
elif status_filter == "Erreurs uniquement":
    df = df[df["has_error"]]

# Filtrer par mode
if mode_filter:
    df = df[df["prompt_mode"].isin(mode_filter)]

# ✅ NOUVEAU: Filtrer par feedback
if feedback_filter == "👍 Positif uniquement":
    df = df[df["user_feedback"] == "thumbs_up"]
elif feedback_filter == "👎 Négatif uniquement":
    df = df[df["user_feedback"] == "thumbs_down"]
elif feedback_filter == "Sans feedback":
    df = df[df["user_feedback"].isna()]

# KPIs
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.metric("📝 Total requêtes", len(df))

with col2:
    success_rate = (len(df[~df["has_error"]]) / len(df) * 100) if len(df) > 0 else 0
    st.metric("✅ Taux succès", f"{success_rate:.1f}%")

with col3:
    avg_time = df["execution_time_seconds"].mean() if len(df) > 0 else 0
    st.metric("⏱️ Temps moyen", f"{avg_time:.2f}s")

with col4:
    avg_docs = df["reranked_docs_count"].mean() if len(df) > 0 else 0
    st.metric("📚 Docs moyens", f"{avg_docs:.1f}")

with col5:
    avg_rerank = (
        df["avg_rerank_score"].mean() if df["avg_rerank_score"].notna().any() else 0
    )
    st.metric("⭐ Score rerank", f"{avg_rerank:.3f}")

with col6:
    # ✅ NOUVEAU: Taux de feedback
    feedback_count = df["user_feedback"].notna().sum()
    feedback_rate = (feedback_count / len(df) * 100) if len(df) > 0 else 0
    st.metric("💬 Taux feedback", f"{feedback_rate:.1f}%")

with col7:
    # ✅ NOUVEAU: Satisfaction
    thumbs_up = (df["user_feedback"] == "thumbs_up").sum()
    thumbs_down = (df["user_feedback"] == "thumbs_down").sum()
    satisfaction = (
        (thumbs_up / (thumbs_up + thumbs_down) * 100)
        if (thumbs_up + thumbs_down) > 0
        else 0
    )
    st.metric("😊 Satisfaction", f"{satisfaction:.1f}%")

# ✅ NOUVEAU: Section Feedback
st.subheader("💬 Analyse des Feedbacks")

if df["user_feedback"].notna().any():
    col1, col2 = st.columns(2)

    with col1:
        # Graphique en camembert des feedbacks
        feedback_counts = df["user_feedback"].value_counts()

        fig_feedback = go.Figure(
            data=[
                go.Pie(
                    labels=["👍 Positif", "👎 Négatif"],
                    values=[
                        feedback_counts.get("thumbs_up", 0),
                        feedback_counts.get("thumbs_down", 0),
                    ],
                    marker=dict(colors=["#28a745", "#dc3545"]),
                )
            ]
        )
        fig_feedback.update_layout(title="Répartition des feedbacks")
        st.plotly_chart(fig_feedback, use_container_width=True)

    with col2:
        # Feedback par mode
        if df["prompt_mode"].notna().any() and df["user_feedback"].notna().any():
            feedback_by_mode = (
                df[df["user_feedback"].notna()]
                .groupby(["prompt_mode", "user_feedback"])
                .size()
                .reset_index(name="count")
            )

            fig_feedback_mode = px.bar(
                feedback_by_mode,
                x="prompt_mode",
                y="count",
                color="user_feedback",
                title="Feedbacks par mode de prompt",
                labels={
                    "prompt_mode": "Mode",
                    "count": "Nombre",
                    "user_feedback": "Feedback",
                },
                color_discrete_map={
                    "thumbs_up": "#28a745",
                    "thumbs_down": "#dc3545",
                },
            )
            st.plotly_chart(fig_feedback_mode, use_container_width=True)
        else:
            st.info("Pas assez de données pour afficher le graphique par mode")

    # Évolution des feedbacks dans le temps
    if len(df[df["user_feedback"].notna()]) > 0:
        df_feedback = df[df["user_feedback"].notna()].copy()
        df_feedback["feedback_date"] = pd.to_datetime(
            df_feedback["feedback_timestamp"]
        ).dt.date

        feedback_timeline = (
            df_feedback.groupby(["feedback_date", "user_feedback"])
            .size()
            .reset_index(name="count")
        )

        fig_feedback_timeline = px.area(
            feedback_timeline,
            x="feedback_date",
            y="count",
            color="user_feedback",
            title="Évolution des feedbacks dans le temps",
            labels={
                "feedback_date": "Date",
                "count": "Nombre",
                "user_feedback": "Feedback",
            },
            color_discrete_map={
                "thumbs_up": "#28a745",
                "thumbs_down": "#dc3545",
            },
        )
        st.plotly_chart(fig_feedback_timeline, use_container_width=True)
else:
    st.info("Aucun feedback utilisateur enregistré pour le moment.")

st.divider()

# Distribution des modes de prompt
st.subheader("🎭 Distribution des modes de prompt")

if df["prompt_mode"].notna().any():
    mode_counts = df["prompt_mode"].value_counts()
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_mode_pie = go.Figure(
            data=[
                go.Pie(
                    labels=mode_counts.index,
                    values=mode_counts.values,
                    marker=dict(colors=["#636EFA", "#EF553B", "#00CC96"]),
                )
            ]
        )
        fig_mode_pie.update_layout(title="Répartition des modes")
        st.plotly_chart(fig_mode_pie, use_container_width=True)

    with col2:
        st.write("**Statistiques par mode**")
        for mode in mode_counts.index:
            mode_df = df[df["prompt_mode"] == mode]
            avg_time_mode = mode_df["execution_time_seconds"].mean()
            st.write(f"**{mode.capitalize()}**")
            st.write(f"- Requêtes : {len(mode_df)}")
            st.write(f"- Temps moyen : {avg_time_mode:.2f}s")

            if mode_df["avg_rerank_score"].notna().any():
                avg_score_mode = mode_df["avg_rerank_score"].mean()
                st.write(f"- Score moyen : {avg_score_mode:.3f}")

            # ✅ NOUVEAU: Satisfaction par mode
            mode_thumbs_up = (mode_df["user_feedback"] == "thumbs_up").sum()
            mode_thumbs_down = (mode_df["user_feedback"] == "thumbs_down").sum()
            if mode_thumbs_up + mode_thumbs_down > 0:
                mode_satisfaction = (
                    mode_thumbs_up / (mode_thumbs_up + mode_thumbs_down) * 100
                )
                st.write(f"- Satisfaction : {mode_satisfaction:.1f}%")

            st.write("")

# Visualisations
st.subheader("📈 Visualisations")

col1, col2 = st.columns(2)

with col1:
    # Requêtes par jour
    queries_by_date = df.groupby("date").size().reset_index(name="count")
    fig_timeline = px.line(
        queries_by_date,
        x="date",
        y="count",
        title="Requêtes par jour",
        labels={"date": "Date", "count": "Nombre de requêtes"},
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with col2:
    # Temps d'exécution par mode
    if df["prompt_mode"].notna().any() and df["execution_time_seconds"].notna().any():
        fig_time_mode = px.box(
            df[df["prompt_mode"].notna()],
            x="prompt_mode",
            y="execution_time_seconds",
            title="Temps d'exécution par mode",
            labels={"prompt_mode": "Mode", "execution_time_seconds": "Temps (s)"},
            color="prompt_mode",
        )
        st.plotly_chart(fig_time_mode, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Distribution des scores
    all_scores = []
    for scores_list in df["rerank_scores_parsed"].dropna():
        all_scores.extend(scores_list)

    if all_scores:
        fig_scores = px.histogram(
            x=all_scores,
            title="Distribution des scores de rerank",
            labels={"x": "Score de rerank", "y": "Fréquence"},
            nbins=50,
        )
        fig_scores.add_vline(
            x=0.5, line_dash="dash", line_color="red", annotation_text="Seuil 0.5"
        )
        st.plotly_chart(fig_scores, use_container_width=True)

with col2:
    # Score moyen par mode
    if df["prompt_mode"].notna().any() and df["avg_rerank_score"].notna().any():
        mode_score_avg = (
            df.groupby("prompt_mode")["avg_rerank_score"].mean().reset_index()
        )
        fig_score_mode = px.bar(
            mode_score_avg,
            x="prompt_mode",
            y="avg_rerank_score",
            title="Score de rerank moyen par mode",
            labels={"prompt_mode": "Mode", "avg_rerank_score": "Score moyen"},
            color="prompt_mode",
        )
        st.plotly_chart(fig_score_mode, use_container_width=True)

# Évolution de l'utilisation des modes
st.subheader("📅 Évolution de l'utilisation des modes")

if df["prompt_mode"].notna().any():
    mode_timeline = df.groupby(["date", "prompt_mode"]).size().reset_index(name="count")
    fig_mode_timeline = px.area(
        mode_timeline,
        x="date",
        y="count",
        color="prompt_mode",
        title="Utilisation des modes dans le temps",
        labels={"date": "Date", "count": "Nombre de requêtes", "prompt_mode": "Mode"},
    )
    st.plotly_chart(fig_mode_timeline, use_container_width=True)

# Top Sources
st.subheader("📚 Top Sources")

all_sources = []
for sources_json in df["sources"].dropna():
    try:
        sources = json.loads(sources_json)
        all_sources.extend([Path(s).name for s in sources])
    except:
        pass

if all_sources:
    from collections import Counter

    top_sources = Counter(all_sources).most_common(10)
    fig_sources = px.bar(
        x=[s[1] for s in top_sources],
        y=[s[0] for s in top_sources],
        orientation="h",
        labels={"x": "Nombre d'utilisations", "y": "Source"},
    )
    fig_sources.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_sources, use_container_width=True)

# Requêtes récentes
st.subheader("🕒 Requêtes récentes")

# ✅ NOUVEAU: Ajouter colonne feedback
display_df = df[
    [
        "id",
        "timestamp",
        "user_query",
        "prompt_mode",
        "execution_time_seconds",
        "avg_rerank_score",
        "user_feedback",
        "has_error",
    ]
].copy()

display_df.columns = [
    "ID",
    "Date",
    "Question",
    "Mode",
    "Temps (s)",
    "Score rerank",
    "Feedback",
    "Erreur",
]

display_df["Date"] = display_df["Date"].dt.strftime("%d/%m/%Y %H:%M:%S")
display_df["Temps (s)"] = display_df["Temps (s)"].round(2)
display_df["Score rerank"] = display_df["Score rerank"].round(4)
display_df["Erreur"] = display_df["Erreur"].map({True: "❌", False: "✅"})
display_df["Mode"] = display_df["Mode"].fillna("N/A")

# ✅ NOUVEAU: Formatter la colonne feedback
display_df["Feedback"] = (
    display_df["Feedback"].map({"thumbs_up": "👍", "thumbs_down": "👎"}).fillna("-")
)

st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)

# Détail d'une requête
st.subheader("🔍 Détail d'une requête")

query_id = st.number_input("ID de la requête", min_value=1, value=1, step=1)

if st.button("Voir détail"):
    query_detail = df[df["id"] == query_id]

    if len(query_detail) == 0:
        st.error(f"❌ Requête {query_id} introuvable")
    else:
        q = query_detail.iloc[0]

        st.write(f"**Date:** {q['timestamp']}")
        st.write(f"**Question:** {q['user_query']}")
        st.write(
            f"**Mode de prompt:** {q['prompt_mode'] if pd.notna(q['prompt_mode']) else 'N/A'}"
        )

        # ✅ NOUVEAU: Afficher le feedback
        if pd.notna(q["user_feedback"]):
            feedback_emoji = "👍" if q["user_feedback"] == "thumbs_up" else "👎"
            feedback_text = (
                "Positif" if q["user_feedback"] == "thumbs_up" else "Négatif"
            )
            st.write(f"**Feedback:** {feedback_emoji} {feedback_text}")
            if pd.notna(q["feedback_timestamp"]):
                st.write(f"**Date du feedback:** {q['feedback_timestamp']}")
        else:
            st.write("**Feedback:** Aucun")

        if q["has_error"]:
            st.error(f"**Erreur:** {q['error']}")
        else:
            import sqlite3

            conn = sqlite3.connect(logger.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_queries WHERE id = ?", (int(query_id),))
            full_q = dict(cursor.fetchone())
            conn.close()

            st.success(f"**Réponse:** {full_q['final_answer']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📥 Docs récupérés", full_q["retrieved_docs_count"])

            with col2:
                st.metric("📊 Docs après rerank", full_q["reranked_docs_count"])

            with col3:
                if q["avg_rerank_score"] is not None:
                    st.metric("⭐ Score moyen", f"{q['avg_rerank_score']:.4f}")

            if full_q["sources"] and full_q["rerank_scores"]:
                try:
                    sources = json.loads(full_q["sources"])
                    scores = json.loads(full_q["rerank_scores"])

                    st.write("**📚 Sources avec scores:**")
                    sources_df = pd.DataFrame(
                        {
                            "Rang": range(1, len(sources) + 1),
                            "Source": [Path(s).name for s in sources],
                            "Score": [
                                f"{s:.4f}" if s is not None else "N/A" for s in scores
                            ],
                        }
                    )
                    st.dataframe(sources_df, use_container_width=True, hide_index=True)
                except:
                    pass

# Export
st.subheader("💾 Export")

if st.button("📥 Télécharger CSV complet"):
    export_df = df.copy()
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Télécharger",
        data=csv,
        file_name=f"rag_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
