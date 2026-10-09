# Rapport d'évaluation de l'assistant « Chat de Calais »

Date : 9 octobre 2026

# 1. Objet et synthèse

Ce rapport rend compte de l'évaluation de la chaîne de recherche et de réponse de « Chat de Calais », assistant qui répond à des questions portant sur les notes de la DDTM du Pas-de-Calais à partir de l'API Albert. L'évaluation visait à mesurer la capacité du système à retrouver la bonne source, à écarter les questions sans rapport avec le corpus et à produire des réponses fidèles, puis à arrêter les paramètres de la configuration par défaut.

Les conclusions sont les suivantes. La configuration de référence retrouve la bonne source pour les 49 questions documentées et rejette les 6 questions hors sujet, avec un MRR de 0,97. La recherche hybride (vectorielle et BM25) est indispensable, car la recherche vectorielle seule ne réussit que dans 18 % des cas. Le rerank Albert assure à la fois le classement des sources (MRR de 0,97 contre 0,48 sans lui) et le rejet des questions hors sujet. HyDE n'apporte aucun gain mesurable dès lors que BM25 est présent, tout en multipliant par environ quarante le temps de recherche (15 s contre 0,4 s), et il a été désactivé. Le seuil de pertinence a été porté de 0,01 à 0,03 après calibrage, et le dédoublonnage combiné supprime tous les passages en double dans les réponses. Une relecture d'une trentaine de chiffres cités n'a révélé aucune erreur, mais le jeu de questions, rédigé à partir des documents eux-mêmes, avantage la recherche lexicale, ce qui limite la portée de ces résultats.

# 2. Méthode

## Corpus et base de test

Le corpus comprend 130 fichiers issus des réunions bilatérales entre la DDTM et le préfet en 2024. Il contient de nombreux doublons : une même note existe parfois sous forme de brouillon ODT et de version validée en PDF, et certains contenus sont repris d'une réunion à l'autre. La base de test a été réindexée avec les corrections du jour. Elle compte 855 passages, dont 199 issus de l'OCR. Grâce au cache OCR, la réindexation prend une vingtaine de secondes.

## Jeu de questions

Le fichier `evaluation/questions.yaml` rassemble 55 questions rédigées à partir de la lecture des documents. Quarante-neuf d'entre elles ont leur réponse dans un document connu : chiffres précis, références exactes (un numéro NOR, par exemple), questions de fond, et contenus issus de l'OCR (un chiffre lu sur une carte, un courriel). Les 6 autres sont hors sujet (tarte au sucre, coupe du monde, imprimante, cours de bourse, capitale de l'Australie, piscine de Lens) et la bonne réponse y est l'absence de passage pertinent. La source attendue est désignée par un fragment de nom de fichier, et toute version du document est acceptée.

## Indicateurs

Le succès est atteint lorsqu'une source attendue figure parmi les passages retenus, ou, pour une question hors sujet, lorsqu'aucun passage n'est retenu. Le rappel de la recherche mesure la présence de la source attendue parmi les candidats avant le rerank. Le MRR est la moyenne de l'inverse du rang de la première bonne source. Le rejet désigne la part des questions hors sujet correctement écartées. Sont également relevés le nombre moyen de passages, le nombre de documents distincts, le nombre de doublons par réponse (passages quasi identiques à un passage mieux classé, selon un critère de mesure fixe pour toutes les variantes) et la durée de la recherche par question, rédaction de la réponse exclue.

## Variantes

La configuration de référence combine une recherche hybride (vectorielle et BM25, fusion RRF), le rerank Albert, un dédoublonnage (Jaccard ou inclusion, seuil de 0,85, au moins 80 empreintes), un plafond de 2 passages par document, 5 passages au plus, un score minimum de 0,03, HyDE désactivé et un en-tête contextuel. Chaque variante retire ou modifie un seul élément de cette référence.

# 3. Résultats

## Tableau principal

| Variante | Succès (documentées) | Rejet (hors sujet) | MRR | Passages moyens | Documents distincts | Doublons par réponse | Durée (s) |
|---|---|---|---|---|---|---|---|
| Référence (sans HyDE, seuil 0,03) | 1,00 | 1,00 | 0,97 | 4,00 | 3,02 | 0,00 | 0,39 |
| Avec HyDE | 1,00 | 1,00 | 0,98 | 4,02 | 3,10 | 0,00 | 15,12 |
| Sans dédoublonnage | 1,00 | 1,00 | 0,96 | 4,25 | 3,24 | 1,05 | 0,33 |
| Dédoublonnage par Jaccard seul | 1,00 | 1,00 | 0,98 | 4,09 | 3,02 | 0,38 | 0,35 |
| Sans plafond par document | 1,00 | 1,00 | 0,97 | 4,33 | 2,76 | 0,00 | 0,36 |
| Sans recherche hybride | 0,18 | 1,00 | 0,18 | 0,60 | 0,55 | 0,00 | 0,35 |
| Sans rerank | 0,92 | 0,00 | 0,48 | 5,00 | 4,35 | 0,00 | 0,11 |
| Sans en-tête contextuel | 1,00 | 1,00 | 0,93 | 3,93 | 3,06 | 0,00 | 0,32 |

Le nombre de documents distincts plus élevé sans dédoublonnage est trompeur : les doublons proviennent de fichiers différents (copies d'une réunion à l'autre) et sont donc comptés comme des documents distincts alors que leur contenu est identique.

Une première série de mesures, réalisée avec HyDE et l'ancien seuil de 0,01, avait donné un succès de 1,00, un MRR de 0,98 et une durée de 14,1 s, mais un rejet de 0,83 seulement, une question hors sujet laissant passer un passage. Sans HyDE, la même configuration atteignait un rejet de 1,00 en 0,32 s. Cette série a aussi montré que HyDE améliore la recherche vectorielle seule (succès de 0,55 contre 0,24), mais devient inutile dès que BM25 est présent.

## Seuil de pertinence

Les scores du rerank Albert sont compris entre 0 et 1. Une simulation sur les mêmes passages montre que, sans HyDE, le succès des questions documentées reste à 1,00 pour tous les seuils de 0,005 à 0,3, tandis que le rejet des questions hors sujet passe de 0,00 sans seuil à 1,00 dès 0,005. Le meilleur score obtenu par une question hors sujet est au plus de 0,0043 (piscine de Lens), les autres se situant entre 0,0000 et 0,0007. À l'inverse, le meilleur score de la bonne source d'une question documentée est d'au moins 0,38 à 0,45 selon la série, pour une médiane de 0,999. L'écart est donc considérable, et le seuil peut être placé loin des deux distributions. Avec HyDE et un seuil de 0,01, une question hors sujet dépassait néanmoins le seuil, ce qui explique le rejet imparfait de la première série.

## Documents similaires

Sur l'ensemble du corpus, le critère combiné (Jaccard, ou inclusion d'un passage dans l'autre au-delà du seuil de taille) détecte 179 paires de passages quasi identiques entre versions d'une même note, contre 107 pour Jaccard seul au seuil de 0,85, soit 67 % de plus ; l'inclusion sans seuil de taille en compterait 220, mais avec de faux doublons. Dans un échantillon contrôlé à la main, les paires ajoutées étaient de véritables doublons, par exemple la même note sur le trait de côte conservée sous deux noms. Le seuil de taille (80 empreintes, soit environ 85 mots) évite de prendre un simple bloc d'adresse pour un doublon. Dans les réponses, on relève 1,05 doublon par réponse sans dédoublonnage, 0,38 avec Jaccard seul et aucun avec le critère combiné. Le plafond par document, de son côté, fait passer le nombre de documents distincts de 2,76 à 3,02, sans coût sur le succès.

## Recherche

La recherche vectorielle seule est faible (18 % de succès, 29 % sans en-tête contextuel), et BM25 porte l'essentiel du rappel. Le rerank remet la bonne source en tête (MRR de 0,97 contre 0,48 sans rerank) et assure seul le rejet des questions hors sujet : sans lui, aucune question hors sujet n'est écartée. L'en-tête contextuel pénalise légèrement la recherche vectorielle seule (rappel de 0,45 sans en-tête), mais améliore la chaîne complète (MRR de 0,97 contre 0,93), car BM25 et le rerank exploitent le titre, la date et le statut du document. Il est donc conservé.

# 4. Qualité des réponses générées

La génération a été lancée sur les 55 questions avec la configuration de référence, puis relue par l'assistant de développement. Les 49 questions documentées ont reçu une réponse comportant des citations numérotées [n]. Les 6 questions hors sujet n'ont donné lieu à aucune génération et ont renvoyé le message « aucun passage suffisamment pertinent ».

Une trentaine de chiffres et de références ont été confrontés aux documents, et tous se sont révélés exacts. Il s'agit notamment de 348 passages à niveau répertoriés, 39 conventions, le décret n° 2021-396, la référence NOR TREL2332413J, 37,1 M€ d'engagements ANRU en 2023, 63 % de l'objectif NPNRU au 1er octobre, 174 logements reconstitués, un objectif LLS ramené à 1 972, 92 % des autorisations d'engagement Anah, 4 442 logements PDC Habitat dans la CA du Boulonnais (lus par OCR sur une carte), 1 250 nids et 237,5 ETP.

Deux défauts mineurs ont été relevés. Certains ajouts ne sont pas sourcés : l'année « 2024 » a été ajoutée à la date de l'avis du CNPN que la note donne sans année, et la date d'une battue n'a pu être vérifiée. Par ailleurs, des formules de clôture superflues (« Pour toute question supplémentaire... ») apparaissent en fin de réponse.

# 5. Décisions prises

Les décisions suivantes sont appliquées dans `parametres.yaml`.

| Paramètre | Décision | Justification |
|---|---|---|
| HyDE | Désactivé par défaut | Temps de recherche divisé par environ quarante (de 15 s à 0,4 s), sans perte mesurée |
| Score minimum | 0,03 (au lieu de 0,01) | Environ sept fois le pire score hors sujet observé, très en dessous des bonnes réponses |
| Alerte de faible pertinence | 0,1, pastille orange alignée sur cette valeur | Distinguer les réponses fragiles sans rejeter de bonne source |
| Dédoublonnage | Critère combiné, taille minimale de 80 empreintes | Aucun doublon dans les réponses, sans faux positif sur les blocs courts |
| En-tête contextuel | Conservé | Meilleur MRR de la chaîne complète (0,97 contre 0,93) |
| Plafond par document | 2 passages, conservé | Davantage de documents distincts par réponse (3,02 contre 2,76) |

# 6. Limites

Les questions ont été rédigées à partir des documents, avec leur vocabulaire, ce qui avantage BM25. Des questions formulées par des agents, avec des synonymes et des paraphrases, pourraient donner plus de poids à la recherche par le sens et à HyDE, dont l'inutilité constatée ici ne doit donc pas être tenue pour définitive. Le jeu ne compte que 49 questions documentées, et plusieurs variantes atteignent un succès de 100 %, de sorte que cet indicateur ne les départage plus : seuls le MRR et les doublons permettent de distinguer les configurations, avec des écarts parfois faibles.

Le succès est mesuré au niveau du document et non de l'exactitude de la réponse, laquelle a été contrôlée à part, par relecture, sur un échantillon. La génération et HyDE ne sont pas déterministes (température de 0,1), si bien que deux exécutions peuvent différer légèrement. Enfin, un seul rédacteur a écrit les questions, ce qui expose le jeu à un biais de formulation propre à une personne.

# 7. Pistes

Plusieurs prolongements sont envisageables. Il convient d'abord de faire rédiger 30 à 50 questions par des agents afin de tester la recherche sur des formulations indépendantes du vocabulaire des documents. Un autre modèle d'embeddings, ou des passages plus courts, pourrait ensuite renforcer la recherche par le sens, aujourd'hui peu efficace. Les formules de clôture superflues peuvent être supprimées par une consigne de prompt. L'évaluation de la fidélité gagnerait à être outillée, en vérifiant automatiquement que chaque chiffre de la réponse figure dans les passages cités. Enfin, l'évaluation devra être refaite à chaque évolution notable du corpus.

# 8. Reproduire l'évaluation

Les commandes suivantes s'exécutent depuis la racine du projet. Les résultats détaillés sont écrits en CSV dans `evaluation/resultats/`.

```
python evaluate.py                                   # configuration de référence
python evaluate.py --no-dedup                        # sans dédoublonnage
python evaluate.py --no-hybrid                       # sans recherche hybride
python evaluate.py --no-rerank                       # sans rerank
python evaluate.py --doublons-taille-min 1000000     # Jaccard seul
python evaluate.py --max-per-doc 0                   # sans plafond par document
USE_HYDE=true python evaluate.py                     # avec HyDE (variable d'environnement)
python evaluate.py --seuils 0,0.005,0.01,0.02,0.05,0.1   # simulation de seuils
python evaluate.py --generate                        # génération des réponses
python evaluate.py --base DOSSIER                    # évaluer une autre base
```
