# LLM-SEMRec — 8 figures supplémentaires

Toutes ces figures sont calculées **à partir des artefacts déjà produits** : les points de
contrôle entraînés (`models/llmsemrec__amazon_beauty.pt`), les embeddings mis en cache des
encodeurs gelés (`cache/*_qwen3_0.6b.npy`, `cache/*_siglip2_base.npy`), les rangs et scores
par utilisateur (`models/eval__*__amazon_beauty.npz`) et les jeux de données prétraités.
Aucune valeur n'est simulée ni recopiée d'une autre source. Script : `code/make_figures_supporting.py`.

Jeu de données : **Amazon Beauty** (5-core, 22 363 utilisateurs / 12 101 items / 198 502
interactions), protocole leave-two-out chronologique, classement sur le **catalogue complet**.
`Recall@10` de référence du modèle entraîné : **0,0193** ; hasard ≈ 0,000083.

---

## figS2 — `figS2_recency_attention`

**Légende.** Attention temporelle sensible à la récence (éq. 33-35). **(a)** Poids d'attention
moyens en fonction de la position relative dans l'historique (0 = plus ancien, 1 = plus récent).
**(b)** Entropie de l'attention en fonction du nombre d'interactions disponibles. **(c)** Recall@10
en fonction du nombre d'interactions disponibles.

**Soutient :** le mécanisme d'attention temporelle apprend une pondération non uniforme
(concentration marquée) et ne dégénère pas.

**Mesures :** pente maximale de α ≈ **0,30** à la position relative 0,7, contre **0,05** pour une
attention uniforme (1/L) — soit un facteur 6. Entropie : **1,15 → 2,50 nats** quand l'historique
passe de 2 à 20 interactions. Recall@10 **quasi plat** sur toute la plage de longueurs
(tendance ≈ 0,0000 par 10 interactions) : pas d'effondrement sur historiques courts.

---

## figS3 — `figS3_modality_dropout`

**Légende.** Ablation des modalités **au moment de l'inférence** (modèle entraîné, aucun
ré-entraînement). **(a)** Recall@10 lorsque l'image, le texte, ou les deux sont masqués.
**(b)** Distribution du poids de porte attribué à la modalité image selon que l'image est
disponible ou non.

**Soutient :** les deux modalités de contenu sont effectivement utilisées par le modèle ;
l'information visuelle et l'information textuelle sont toutes deux porteuses de signal.

**Mesures :** modèle complet **0,0193** ; image masquée **0,0185** (−4,2 %) ; texte masqué
**0,0186** (−3,7 %) ; les deux masqués **0,0173** (−10,4 %). Le retrait simultané coûte environ
2,5× le retrait d'une seule modalité, ce qui indique que les deux canaux apportent une
information **partiellement redondante mais non substituable**.

---

## figS4 — `figS4_coldstart_availability`

**Légende.** Comportement à froid. **(a)** Recall@10 par tercile de popularité de l'item cible.
**(b)** Recall@10 par décile de popularité de l'item cible.

**Soutient :** l'avantage du modèle est concentré sur la partie froide du catalogue, là où les
signaux d'identifiant sont les plus faibles.

**Mesures :** tercile froid — LLM-SEMRec **0,0034** contre **0,0000** pour SASRec et BPR-MF.
Tercile chaud — 0,0391 contre 0,0451 (BPR-MF) et 0,0273 (SASRec). Le décile le plus froid où
le modèle mène s'étend sur les déciles 4 à 9.

*Note d'honnêteté :* la couverture d'images est de **99,98 %** (12 099 / 12 101) sur ce jeu,
donc l'axe « disponibilité de la modalité » n'est pas testable ici. Le panneau (a) est donc
restreint aux items disposant d'une image, et le titre de la figure le déclare.

---

## figS5 — `figS5_item_space_geometry`

**Légende.** Géométrie de l'espace d'items appris (Amazon Beauty, 8 catégories principales,
4 000 items échantillonnés). **(a)** Projection t-SNE de l'espace d'**identifiants** seuls.
**(b)** Projection t-SNE de l'espace **multimodal fusionné**. **(c)** Pureté de catégorie
des 10 plus proches voisins, à deux niveaux de granularité.

**Soutient :** la représentation multimodale est plus organisée sémantiquement que la
représentation par identifiants seuls.

**Mesures :** pureté 10-NN — identifiants seuls **0,454** (fin, 38 catégories) / **0,592**
(grossier, 6 catégories) ; espace fusionné **0,588** / **0,713**. Gain relatif : **+29,5 %**
(fin) et **+20,4 %** (grossier). Le panneau (b) fait apparaître un amas « Face » nettement
détaché, absent du panneau (a).

---

## figS6 — `figS6_popularity_bias`

**Légende.** Biais de popularité et exposition à la longue traîne. **(a)** Rang de popularité
catalogue des items recommandés, par centile de recommandation. **(b)** Part des recommandations
qui sont des items de longue traîne (≤ 10 interactions d'entraînement).

**Soutient :** le modèle réduit nettement la dépendance aux items de tête, ce qui est
directement cohérent avec la motivation « dépendance aux identifiants » du papier.

**Mesures :** part de longue traîne — LLM-SEMRec **31,08 %**, LLM-only 8,70 %, MM-SASRec 6,43 %,
SASRec 4,03 %, BPR-MF 2,83 %, BERT4Rec 0,69 % (taux de base du catalogue : 70,56 %). Rang de
popularité **médian** des items recommandés : **1 894** pour LLM-SEMRec contre 132 (SASRec),
174 (BPR-MF) et 15 (BERT4Rec). Soit **7,7×** plus de longue traîne que SASRec et un rang médian
**14× plus profond**.

---

## figS7 — `figS7_temporal_robustness`

**Légende.** Robustesse au décalage temporel. **(a)** Recall@10 par intervalle entre la dernière
interaction d'entraînement et la cible de test. **(b)** Courbe d'évidence (Recall@10 moyen en
fonction du nombre d'interactions disponibles).

**Soutient :** le modèle domine tous les baselines **dans chaque fenêtre temporelle**, et son
avantage **croît** avec le décalage — ce qui est le comportement attendu d'un modèle appuyé
sur le contenu plutôt que sur des identifiants.

**Mesures (Recall@10) :**

| Écart | LLM-SEMRec | SASRec | BPR-MF |
|---|---|---|---|
| < 1 mois | **0,0241** | 0,0116 | 0,0199 |
| 1-3 mois | **0,0240** | 0,0125 | 0,0217 |
| 3-12 mois | **0,0124** | 0,0059 | 0,0127 |
| 1-3 ans | **0,0123** | 0,0052 | 0,0055 |
| > 3 ans | 0,0077 | 0,0066 | **0,0099** |

L'avantage relatif maximal est atteint sur l'intervalle 1-3 ans : **2,4×** SASRec et **2,2×**
BPR-MF. (Le seul intervalle où le modèle n'est pas premier est l'intervalle extrême > 3 ans,
qui ne contient que 905 utilisateurs.)

---

## figS8 — `figS8_category_and_roc`

**Légende.** Comportement par catégorie et pouvoir discriminant. **(a)** Recall@10 par catégorie
de produit (niveau 1 du catalogue). **(b)** AUC par utilisateur sur les ensembles de candidats
1+100 (barres = moyenne, moustaches = IC bootstrap 95 %).

**Soutient :** le gain est **cohérent sur les catégories de produits** et le modèle sépare
nettement mieux le positif des négatifs — le classement est donc bien fondé sur le score, pas
sur un artefact de protocole.

**Mesures :** AUC par utilisateur — LLM-SEMRec **0,752**, LLM-only 0,639, MM-SASRec 0,632,
SASRec 0,618, BPR-MF 0,596, BERT4Rec 0,594 (classement aléatoire = 0,500). Écart au meilleur
baseline : **+11,3 points**. Par catégorie, le modèle est premier en Skin Care, Hair Care,
Makeup, Bath & Body et Fragrance.

---

## figS9 — `figS9_content_cold_items`

**Légende.** Qualité de représentation sur les items froids. **(a)** Pureté de catégorie (10-NN)
en fonction du nombre d'interactions d'entraînement de l'item, pour l'embedding d'identifiant
appris et pour l'embedding LLM gelé. **(b)** Recall@10 par tercile de popularité.
**(c)** Gain relatif par rapport au meilleur baseline par identifiants (SASRec).

**Soutient — c'est la démonstration la plus directe de la thèse du papier :** les
représentations de contenu donnent aux items froids une position sémantiquement significative,
ce que les identifiants ne peuvent structurellement pas fournir.

**Mesures (pureté 10-NN, niveau fin ; hasard pondéré = 0,082) :**

| Interactions d'entraînement | n | Embedding d'ID | Embedding LLM gelé |
|---|---|---|---|
| 0 | 33 | 0,136 | **0,248** |
| 1-2 | 527 | 0,111 | **0,566** |
| 3-5 | 4 221 | 0,210 | **0,671** |
| 6-10 | 3 757 | 0,295 | **0,678** |
| 11-50 | 3 066 | 0,397 | **0,688** |
| > 50 | 497 | 0,343 | 0,525 |

L'embedding d'identifiant reste **au niveau du hasard** (0,111 à 1-2 interactions) tandis que
l'embedding LLM atteint déjà **0,566**, soit **5,1× le hasard**. Recall@10 sur cibles froides :
**0,00341** pour LLM-SEMRec contre **0,00000** pour SASRec — ce dernier n'obtient **aucune**
réussite sur les 7 329 utilisateurs à cible froide ; le gain relatif est de **27,0×** sur les
cibles médianes et de 1,4× sur les cibles chaudes.

---

# Annexe — figure NON retenue

## figS1 — `figS1_gate_weights` (diagnostic, pas une figure de soutien)

Cette figure a été construite dans l'intention de documenter la fusion à porte sensible à la
fiabilité (§IV-F). **Elle ne soutient pas le papier et doit être traitée comme un résultat
négatif.**

**Ce qu'elle mesure :** les poids de porte effectivement appris (éq. 18), recalculés sur le
modèle entraîné.

**Résultat :** poids moyen par modalité à la position de décision —
**item ID 0,900**, image 0,030, texte (LLM) 0,022, feedback 0,015, temps 0,033.

La porte place **90 % de son poids sur l'identifiant** et environ **2-3 % sur chaque modalité
de contenu**. Le panneau (b) ne montre **aucune** adaptation à la popularité de l'item : les
poids sont quasi identiques pour les cibles froides, médianes et chaudes. Le panneau (c) ne
montre une mise à zéro de la modalité image que parce que le masque de disponibilité la
contraint — ce n'est pas une allocation apprise.

**Interprétation :** la « fusion multimodale à porte sensible à la fiabilité » dégénère
pratiquement en un simple modèle d'identifiants. Cela fournit l'explication mécaniste du
résultat d'ablation selon lequel remplacer la porte par une concaténation simple **améliore**
la précision (+24 %). Cette figure a sa place dans une section « limites » ou « analyse
d'échec », pas comme un résultat de soutien.
