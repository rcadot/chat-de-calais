from pathlib import Path


# Fonction pour formater les sources
def format_sources(sources, scores):
    """
    Formate les sources avec scores en HTML.
    Déduplique les sources et garde le meilleur score.
    """
    if not sources:
        return ""

    # Dédupliquer : garder le meilleur score pour chaque source unique
    sources_dict = {}
    for source, score in zip(sources, scores):
        source_name = Path(source).name

        if source_name not in sources_dict:
            sources_dict[source_name] = score
        else:
            # Garder le meilleur score
            if score and sources_dict[source_name]:
                sources_dict[source_name] = max(sources_dict[source_name], score)
            elif score:
                sources_dict[source_name] = score

    # Trier par score décroissant
    sorted_sources = sorted(
        sources_dict.items(), key=lambda x: x[1] if x[1] else 0, reverse=True
    )

    sources_html = '<div class="sources-section">'
    sources_html += "<strong>📚 Sources consultées :</strong><br><br>"

    for i, (source_name, score) in enumerate(sorted_sources, 1):
        if score:
            percentage = score * 100
            if score >= 0.7:
                score_class = "source-score"
                emoji = "🟢"
            elif score >= 0.5:
                score_class = "source-score source-score-medium"
                emoji = "🟡"
            else:
                score_class = "source-score source-score-low"
                emoji = "🔴"
            score_display = (
                f'<span class="{score_class}">{emoji} {percentage:.0f}%</span>'
            )
        else:
            score_display = '<span class="source-score">⚪ N/A</span>'

        sources_html += (
            f'<div class="source-item">{score_display} <code>{source_name}</code></div>'
        )

    sources_html += "</div>"
    return sources_html
