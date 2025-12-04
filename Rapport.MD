# Rapport d'Analyse : Prédiction du Risque de Défaut de Crédit
CHERESTAL Deborah Nativa 
22008149

![Deborah Nativa Cherestal](![WhatsApp Image 2025-10-30 at 11 50 17](https://github.com/user-attachments/assets/1a60df9d-1939-410d-ba24-a93462ac4deb)
) 
## 1. Introduction

### 1.1 Contexte

Les institutions bancaires jouent un rôle fondamental dans les économies modernes en déterminant l'accès au crédit pour les particuliers et les entreprises. Cette décision repose sur des algorithmes de notation de crédit qui évaluent la probabilité de défaut de paiement d'un emprunteur.

### 1.2 Problématique

Le défi consiste à améliorer les modèles existants de credit scoring pour prédire avec précision la probabilité qu'un individu connaisse des difficultés financières dans les deux années suivant l'octroi d'un crédit. Une prédiction erronée peut entraîner soit des pertes financières importantes pour la banque (faux négatifs), soit le refus injustifié de crédit à des clients solvables (faux positifs).

### 1.3 Objectifs

- Développer un modèle prédictif capable d'identifier les clients à risque de défaut de paiement
- Comparer les performances de plusieurs algorithmes de machine learning
- Optimiser le modèle le plus performant pour maximiser la précision des prédictions

## 2. Données Utilisées

### 2.1 Description du Dataset

Le dataset `cs-training.csv` contient des informations financières et comportementales sur les emprunteurs :

**Dimensions initiales :**
- Nombre d'observations après nettoyage : Variable selon les doublons détectés
- Nombre de variables : Environ 11 colonnes initiales

**Variable cible :**
- `SeriousDlqin2yrs` : Indicateur binaire de détresse financière (0 = pas de défaut, 1 = défaut)

**Variables explicatives principales :**
- `RevolvingUtilizationOfUnsecuredLines` : Taux d'utilisation des lignes de crédit renouvelables
- `Age` : Âge de l'emprunteur
- `NumberOfTime30-59DaysPastDueNotWorse` : Nombre de retards de paiement de 30-59 jours
- `DebtRatio` : Ratio d'endettement mensuel
- `MonthlyIncome` : Revenu mensuel
- `NumberOfOpenCreditLinesAndLoans` : Nombre de lignes de crédit et prêts ouverts
- `NumberOfTimes90DaysLate` : Nombre de retards de paiement supérieurs à 90 jours
- `NumberRealEstateLoansOrLines` : Nombre de prêts immobiliers
- `NumberOfTime60-89DaysPastDueNotWorse` : Nombre de retards de paiement de 60-89 jours
- `NumberOfDependents` : Nombre de personnes à charge

### 2.2 Déséquilibre des Classes

L'analyse de la variable cible révèle un déséquilibre important entre les classes, typique des problèmes de détection de fraude ou de défaut de crédit, où les événements de défaut sont minoritaires.

## 3. Méthodologie

### 3.1 Prétraitement des Données

#### 3.1.1 Nettoyage Initial

**Suppression des doublons :**
Les observations en double ont été identifiées et éliminées pour éviter un biais dans l'entraînement du modèle. Cette étape garantit que chaque client n'est représenté qu'une seule fois.

**Suppression de la colonne indexée :**
La colonne `Unnamed: 0`, identifiée comme un simple index sans valeur prédictive, a été retirée du dataset.

#### 3.1.2 Traitement des Valeurs Manquantes

Deux variables présentaient des valeurs manquantes :
- `MonthlyIncome` : Imputation par la médiane
- `NumberOfDependents` : Imputation par la médiane

**Justification :** La médiane a été choisie plutôt que la moyenne car elle est plus robuste aux valeurs extrêmes, particulièrement pertinent pour les données financières qui présentent souvent des distributions asymétriques.

#### 3.1.3 Traitement des Valeurs Aberrantes

**Gestion des retards de paiement extrêmes :**
Les valeurs supérieures à 90 dans les colonnes de retards de paiement ont été remplacées par la catégorie `VeryHighPastDue`, puis encodées via one-hot encoding. Cette approche permet de capturer un comportement de défaut chronique sans laisser des valeurs extrêmes fausser le modèle.

#### 3.1.4 Normalisation

**StandardScaler appliqué aux variables numériques :**
Les features numériques (à l'exception de la variable cible et des variables encodées) ont été normalisées pour avoir une moyenne de 0 et un écart-type de 1.

**Justification :** Cette transformation est essentielle pour :
- Accélérer la convergence des algorithmes de régression logistique
- Permettre une comparaison équitable des poids des différentes variables
- Améliorer les performances des algorithmes sensibles à l'échelle

### 3.2 Feature Engineering

#### 3.2.1 Création de Variables Indicatrices

Trois nouvelles features binaires ont été créées basées sur le 99ème percentile :
- `High_RevolvingUtilization_Flag` : Utilisation très élevée du crédit renouvelable
- `High_DebtRatio_Flag` : Ratio d'endettement très élevé
- `High_MonthlyIncome_Flag` : Revenu mensuel très élevé

**Justification :** Ces indicateurs permettent au modèle de capturer des comportements extrêmes qui peuvent être des signaux forts de risque (ou de stabilité financière pour le revenu).

### 3.3 Analyse Exploratoire

#### 3.3.1 Distributions des Variables

Des histogrammes ont été générés pour visualiser les distributions des variables normalisées, permettant d'identifier d'éventuelles asymétries ou patterns.

#### 3.3.2 Détection des Outliers

Des boxplots ont révélé la présence de valeurs extrêmes dans plusieurs variables, confirmant la pertinence du choix de la médiane pour l'imputation.

#### 3.3.3 Analyse de Corrélation

Une matrice de corrélation complète a été calculée pour identifier :
- Les relations entre les variables explicatives
- Les corrélations avec la variable cible
- Les problèmes potentiels de multicolinéarité

### 3.4 Modélisation

#### 3.4.1 Division des Données

Le dataset a été divisé selon un ratio 80/20 (entraînement/test) avec stratification sur la variable cible pour maintenir le déséquilibre des classes dans les deux ensembles.

#### 3.4.2 Algorithmes Testés

Trois algorithmes ont été comparés :

1. **Régression Logistique**
   - Modèle linéaire simple et interprétable
   - Baseline de référence
   - Solver : liblinear (adapté aux petits datasets)

2. **Random Forest**
   - Modèle ensembliste robuste aux outliers
   - Capable de capturer des interactions non-linéaires
   - Moins sujet au surapprentissage

3. **Gradient Boosting**
   - Algorithme de boosting performant
   - Construction séquentielle d'arbres
   - Généralement le plus performant sur les problèmes tabulaires

#### 3.4.3 Optimisation par GridSearchCV

Le modèle Gradient Boosting a été sélectionné pour une optimisation approfondie via GridSearchCV avec :
- Validation croisée stratifiée (3 folds)
- Métrique d'optimisation : ROC-AUC (appropriée pour les données déséquilibrées)
- Recherche sur grille pour :
  - `n_estimators` : [100, 200, 300]
  - `learning_rate` : [0.05, 0.1, 0.2]
  - `max_depth` : [3, 4, 5]

**Justification du choix de ROC-AUC :** Cette métrique évalue la capacité du modèle à discriminer entre les classes sans être biaisée par le déséquilibre des données, contrairement à l'accuracy.

## 4. Résultats & Discussion

### 4.1 Métriques de Performance

Les trois modèles ont été évalués selon cinq métriques complémentaires :

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Régression Logistique | À compléter | À compléter | À compléter | À compléter | À compléter |
| Random Forest | À compléter | À compléter | À compléter | À compléter | À compléter |
| Gradient Boosting | À compléter | À compléter | À compléter | À compléter | À compléter |

*Note : Les valeurs exactes dépendent de l'exécution complète du code*

### 4.2 Interprétation des Métriques

**Accuracy :** Proportion de prédictions correctes. Attention : peut être trompeuse en cas de classes déséquilibrées.

**Precision :** Parmi les clients prédits comme à risque, quelle proportion l'est réellement ? Métrique importante pour éviter de refuser du crédit à tort.

**Recall (Sensibilité) :** Parmi les clients réellement à risque, quelle proportion est correctement identifiée ? Crucial pour minimiser les pertes financières.

**F1-Score :** Moyenne harmonique entre precision et recall, offrant un équilibre entre les deux.

**ROC-AUC :** Mesure la capacité du modèle à discriminer entre les classes sur l'ensemble des seuils possibles. Valeur proche de 1 indique une excellente performance.

### 4.3 Analyse des Erreurs

#### 4.3.1 Matrice de Confusion

La matrice de confusion permet d'analyser :
- **Vrais Positifs (TP) :** Clients à risque correctement identifiés
- **Vrais Négatifs (TN) :** Clients sains correctement identifiés
- **Faux Positifs (FP) :** Clients sains incorrectement rejetés (perte d'opportunité)
- **Faux Négatifs (FN) :** Clients à risque non détectés (perte financière)

#### 4.3.2 Trade-off Métier

Dans le contexte bancaire :
- Un **FN** coûte plus cher qu'un **FP** (perte du capital prêté vs perte d'opportunité)
- Il faut privilégier un **Recall élevé** pour minimiser les défauts non détectés
- La **Precision** reste importante pour ne pas rejeter systématiquement les demandes

### 4.4 Importance des Variables

L'analyse des modèles ensemblistes (Random Forest et Gradient Boosting) permet d'identifier les variables les plus prédictives :
- Les historiques de retards de paiement sont typiquement les plus importants
- Le ratio d'utilisation du crédit renouvelable
- L'âge et le ratio d'endettement

## 5. Conclusion

### 5.1 Synthèse

Ce projet a permis de développer et comparer trois modèles de prédiction du risque de crédit. Le preprocessing rigoureux, incluant le traitement des valeurs manquantes, la gestion des outliers et la création de features, a posé les bases d'une modélisation robuste.

### 5.2 Limites du Modèle

1. **Déséquilibre des classes :** Malgré l'utilisation de métriques adaptées, le déséquilibre important peut limiter la capacité du modèle à bien prédire la classe minoritaire.

2. **Données manquantes :** L'imputation par la médiane, bien que robuste, introduit une approximation qui peut affecter la précision.

3. **Linéarité des relations :** La régression logistique suppose des relations linéaires qui peuvent ne pas capturer toute la complexité des comportements financiers.

4. **Absence de dimension temporelle :** Le modèle ne capture pas l'évolution des comportements dans le temps.

5. **Généralisation :** Les performances observées sur le jeu de test peuvent ne pas se reproduire sur de nouvelles données si la distribution change.

### 5.3 Pistes d'Amélioration

1. **Techniques de rééquilibrage :**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Sous-échantillonnage de la classe majoritaire
   - Ajustement des poids de classe dans les algorithmes

2. **Modèles avancés :**
   - XGBoost ou LightGBM pour de meilleures performances
   - Réseaux de neurones pour capturer des interactions complexes
   - Stacking ou blending de plusieurs modèles

3. **Feature engineering avancé :**
   - Création d'interactions entre variables
   - Transformations non-linéaires
   - Agrégations temporelles si données historiques disponibles

4. **Optimisation du seuil de décision :**
   - Ajuster le seuil de classification selon le coût métier des erreurs
   - Utiliser une courbe de coût pour trouver le point optimal

5. **Validation externe :**
   - Tester le modèle sur un dataset complètement indépendant
   - Monitoring des performances en production

6. **Interprétabilité :**
   - Utiliser SHAP values pour expliquer les prédictions individuelles
   - Développer des règles métier interprétables basées sur le modèle

7. **Variables supplémentaires :**
   - Intégrer des données comportementales (transactions, navigation)
   - Enrichir avec des données externes (données macroéconomiques, secteur d'activité)

---

**Date du rapport :** Décembre 2025  
**Dataset :** Credit Scoring Dataset (cs-training.csv)  
**Outils utilisés :** Python, Pandas, Scikit-learn, Seaborn, Matplotlib
