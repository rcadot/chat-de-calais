# assets/dessiner_avatar.py
"""Dessine l'avatar de l'assistant (chat stylisé, couleurs de l'État).

Usage : python assets/dessiner_avatar.py  ->  assets/avatar.png
Le dessin est fait à 4 fois la taille finale puis réduit (anticrénelage).
"""

from pathlib import Path

from PIL import Image, ImageDraw

BLEU = (0, 0, 145)  # bleu France
ROUGE = (225, 0, 15)  # rouge Marianne
BLANC = (255, 255, 255)
TAILLE = 256
ECHELLE = 4


def dessiner(taille: int = TAILLE) -> Image.Image:
    """
    Dessine l'avatar.

    Args:
        taille: Côté de l'image finale en pixels

    Returns:
        Image.Image: Image RGBA
    """
    s = taille * ECHELLE
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def p(x: float, y: float) -> tuple:
        """Coordonnées relatives (0 à 1) vers pixels."""
        return (x * s, y * s)

    # Fond rond bleu
    d.ellipse([p(0, 0), p(1, 1)], fill=BLEU)

    # Oreilles (blanches, intérieur rouge)
    d.polygon([p(0.22, 0.47), p(0.27, 0.16), p(0.47, 0.33)], fill=BLANC)
    d.polygon([p(0.78, 0.47), p(0.73, 0.16), p(0.53, 0.33)], fill=BLANC)
    d.polygon([p(0.27, 0.42), p(0.29, 0.24), p(0.41, 0.34)], fill=ROUGE)
    d.polygon([p(0.73, 0.42), p(0.71, 0.24), p(0.59, 0.34)], fill=ROUGE)

    # Tête
    d.ellipse([p(0.20, 0.28), p(0.80, 0.84)], fill=BLANC)

    # Yeux (amande bleue, reflet blanc)
    for cx in (0.38, 0.62):
        d.ellipse([p(cx - 0.055, 0.49), p(cx + 0.055, 0.58)], fill=BLEU)
        d.ellipse([p(cx - 0.012, 0.505), p(cx + 0.018, 0.535)], fill=BLANC)

    # Nez rouge et bouche
    d.polygon([p(0.46, 0.63), p(0.54, 0.63), p(0.50, 0.68)], fill=ROUGE)
    w = int(0.012 * s)
    d.arc([p(0.42, 0.63), p(0.50, 0.72)], start=20, end=160, fill=BLEU, width=w)
    d.arc([p(0.50, 0.63), p(0.58, 0.72)], start=20, end=160, fill=BLEU, width=w)

    # Moustaches
    for dy in (-0.025, 0.02):
        d.line([p(0.36, 0.665 + dy), p(0.14, 0.64 + 2 * dy)], fill=BLEU, width=w)
        d.line([p(0.64, 0.665 + dy), p(0.86, 0.64 + 2 * dy)], fill=BLEU, width=w)

    return img.resize((taille, taille), Image.LANCZOS)


if __name__ == "__main__":
    out = Path(__file__).with_name("avatar.png")
    dessiner().save(out, optimize=True)
    print(f"{out} ({out.stat().st_size // 1024} Ko)")
