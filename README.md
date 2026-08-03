Harvest Games
Où planter quoi dans le futur ?
Harvest Games est un projet étudiant d’aide à la décision destiné principalement aux chambres d’agriculture, aux collectivités territoriales et aux acteurs souhaitant mieux comprendre les effets du changement climatique sur l’agriculture française.
L’application croise des données climatiques, pédologiques, géographiques et agricoles afin d’explorer l’évolution de la viabilité et du rendement potentiel de trois cultures :
la vigne à raisin de cuve : une culture de tradition française. La France est un acteur majeur dans l’exportation du vin dans le monde
la pomme de terre : la France est le 2e producteur européen derrière l’Allemagne
le pois chiche : c’est une culture mineure actuellement. Cependant, elle est peu gourmande en ressources et pourrait être une culture alternative dans le futur.
Les résultats sont visualisés sur des cartes interactives à une résolution d’environ 12 km, pour les horizons 2054, 2084 et 2100.
12km = une maille imposée par les options à notre disposition pour la cartographie.
[!WARNING]
Harvest Games est une preuve de concept pédagogique. Les résultats produits sont des estimations exploratoires fondées sur des données publiques, des projections climatiques et des modèles statistiques présentant plusieurs limites. Ils ne constituent pas des recommandations agronomiques opérationnelles.

Sommaire
Problématique
Objectifs
Cultures étudiées
Fonctionnalités
Démonstration
Architecture technique
Pipeline de données
Modélisation des données
Machine learning
Sources de données
Technologies
Installation
Configuration
Lancement
Structure du dépôt
Résultats et limites
Perspectives
Équipe
Licence

Problématique
Le changement climatique modifie progressivement les équilibres agricoles français :
hausse des températures moyennes ;
multiplication des épisodes de chaleur extrême ;
évolution de la répartition des précipitations ;
déplacement possible des zones favorables à certaines cultures ;
augmentation de l’incertitude sur les rendements.
Harvest Games cherche à répondre à une question centrale :
Où planter quoi dans le futur ?
L’objectif n’est pas uniquement d’identifier les zones menacées, mais aussi de faire apparaître les territoires qui pourraient devenir favorables à de nouvelles cultures.

Objectifs
Harvest Games vise à :
visualiser l’effet de la hausse des températures sur les cultures actuelles ;
croiser les contraintes climatiques avec les caractéristiques des sols ;
repérer les zones maintenues, perdues ou nouvellement favorables ;
estimer un rendement agricole potentiel à partir de données historiques ;
comparer plusieurs scénarios climatiques futurs ;
proposer un score global d’adéquation entre une culture et les conditions locales ;
sensibiliser aux besoins d’adaptation de l’agriculture française.

Cultures étudiées
Le projet se concentre sur trois cultures aux profils agroclimatiques contrastés.
Culture
Particularités
Vigne à raisin de cuve
Culture majeure pour l’économie et les exportations françaises, sensible aux évolutions du climat et des terroirs
Pomme de terre
Culture fortement exposée aux températures élevées et aux stress hydriques
Pois chiche
Culture peu consommatrice en eau, susceptible de devenir pertinente dans de nouveaux territoires

Ce choix permet d’illustrer des réactions différentes face aux mêmes évolutions climatiques.

Fonctionnalités
1. Analyse de la hausse des températures
La première carte permet de :
sélectionner une culture ;
comparer la situation actuelle aux projections futures ;
explorer les horizons 2054, 2084 et 2100 ;
simuler des hausses de température plus sévères que les projections officielles ;
distinguer les zones viables, critiques et potentiellement fatales pour la culture.
2. Cartographie des opportunités
La deuxième carte croise :
les zones agricoles actuellement cultivées ;
les projections climatiques futures ;
les contraintes pédologiques ;
les besoins propres à chaque culture.
Les mailles sont classées en plusieurs catégories :
zones maintenues : actuellement cultivées et toujours favorables ;
zones perdues : actuellement cultivées mais devenues défavorables ;
nouvelles zones favorables : non cultivées actuellement mais compatibles avec les conditions futures.
3. Projection des rendements
Une troisième approche estime l’évolution possible du rendement agricole à l’échelle départementale, en cohérence avec la granularité des données publiques disponibles.
Les résultats peuvent indiquer :
une hausse potentielle ;
une baisse potentielle ;
l’apparition d’un potentiel de rendement dans un territoire où la culture est aujourd’hui peu ou pas présente.
4. Score global d’adéquation
Un score synthétique croise plusieurs dimensions :
température ;
précipitations ;
rayonnement ou ensoleillement ;
nature du sol ;
compatibilité du sol avec la culture ;
rendement potentiel.
Ce score vise à comparer les zones entre elles et, à terme, à départager plusieurs cultures lorsqu’une même zone leur est favorable.

Démonstration
docs/
├── screenshots/
│   ├── temperature-map.png
│   ├── opportunity-map.png
│   ├── yield-map.png
│   └── score-map.png
└── demo/
    └── harvest-games-demo.mp4
└── tables_schema/
    └── star_schema_harvest_games.png


Architecture technique
Harvest Games couvre l’ensemble du cycle de vie de la donnée :
Sources de données
        │
        ▼
Extraction par API ou import manuel
        │
        ▼
Stockage AWS S3
Bronze → Silver → Gold
        │
        ▼
Transformations ETL
        │
        ▼
Base PostgreSQL Neon
Schéma en étoile
        │
        ├──────────────► Machine learning
        │                   │
        │                   ├── notebooks
        │                   ├── scripts de prédiction
        │                   └── API FastAPI alternative
        │
        ▼
Application Streamlit
Cartes, scénarios et indicateurs
        │
        ▼
Conteneurisation Docker
Composants principaux
AWS S3 : data lake organisé selon une architecture en médaillon ;
ETL Python : extraction, nettoyage, harmonisation, remaillage et chargement ;
Neon PostgreSQL : source de vérité structurée selon un schéma en étoile ;
notebooks de machine learning : entraînement, comparaison et évaluation des modèles ;
FastAPI : exposition d’un modèle alternatif de prédiction ;
Streamlit : interface utilisateur et visualisations cartographiques ;
Docker Compose : exécution reproductible de l’application.

Datasets

Seulement les fichiers raw sont nécessaires pour l'exécution du pipeline, les autres fichiers étant généré par le script de transformation. Certains datasets peuvent être extraits depuis des sources externes via api (Copernicus), d’autres doivent être téléchargées manuellement ou créées.

Les données ne sont pas stockées sur ce dépôt GitHub en raison de leur taille. Les datasets sont accessibles aux emplacements suivants :

- **Raw :** https://drive.google.com/drive/folders/1B1JTUPIQl1FFZ7KzaAovVuCdNIo5iLzd?usp=sharing
- **Bronze :** https://drive.google.com/drive/folders/1V2iXMVCCCiTCWHwGEgBwU3m2sQFov44W?usp=sharing
- **Silver :** https://drive.google.com/drive/folders/1JYAiOiLDO2S27g9hdVxu3z6zJC8o3Czh?usp=sharing
- **Gold :** https://drive.google.com/drive/folders/14sqMnSNRJVBTZP6wK6qhmMuNNZ5197Vl?usp=sharing

 Instructions d'installation locale
- Après téléchargement des fichiers raw, uploadez-les dans un dossier raw sur AWS S3.
- Les datasets raw peuvent aussi être placés dans un dossier local à la racine du projet (dataset/raw)


Liste des datasets raw (données brutes)

* **Données géographiques & cartographiques :**
  * `cultures_selectionnees_2024.gpkg` : Données vectorielles (GeoPackage) des parcelles et types de cultures sélectionnées pour 2024.
  * `departements.geojson` : Limites administratives des départements français (format GeoJSON).

* **Référentiels & Dimensions :**
  * `dim_constraints.csv` : Table de référence des contraintes agronomiques ou environnementales.
  * `dim_coordinates.csv` : Table de référence des coordonnées géographiques (points de grille / stations).

* **Données climatiques Projections (CORDEX) :**
  * `raw_climate_cordex_2m_air_temperature_*.zip` : Données de températures de l'air à 2 mètres.
  * `raw_climate_cordex_mean_precipitation_flux_*.zip` : Flux moyen de précipitations.
  * `raw_climate_cordex_surface_solar_radiation_downwards_*.zip` : Rayonnement solaire incident au sol.

* **Données climatiques Historiques/Mensuelles (ERA5 - NetCDF) :**
  * `raw_era5_monthly_2m_temperature_*.nc` : Historique mensuel des températures à 2 mètres.
  * `raw_era5_monthly_surface_solar_radiation_downwards_*.nc` : Historique mensuel du rayonnement solaire.
  * `raw_era5_monthly_total_precipitation_*.nc` : Historique mensuel du cumul des précipitations.

* **Données agricoles & Sols :**
  * `SAA_2010-2025_provisoires_donnees_departementales.xlsx` : Statistique Agricole Annuelle (surfaces, rendements, productions) par département.
  * `sols_france_agricoles.csv` : Caractéristiques et typologie des sols agricoles français.




Pipeline de données
Les données suivent plusieurs flux thématiques.
Flux géographique et agricole
décompression ;
conversion des formats ;
harmonisation des identifiants ;
rapprochement avec les coordonnées ;
remaillage spatial ;
calcul des surfaces et agrégations.
Flux climatique
récupération des données climatiques ;
nettoyage ;
transformation temporelle ;
remaillage ;
construction des variables de température, précipitations et rayonnement.
Flux de performance
préparation des rendements historiques ;
création des jeux d’entraînement et de test ;
calcul des scores ;
génération des projections futures.
Chargement
Les tables préparées dans la couche Gold de S3 sont chargées dans Neon PostgreSQL afin d’alimenter :
l’application Streamlit ;
les notebooks de machine learning ;
les scripts de prédiction ;
l’API FastAPI.

Modélisation des données
Les tables Neon sont organisées selon un schéma en étoile.
Dimensions principales
dim_constraints : besoins agroclimatiques et pédologiques des cultures ;
dim_coordinates : coordonnées géographiques des mailles ;
dim_soils : caractéristiques des sols, départements et régions.
Sources et faits
données climatiques historiques et futures : https://cds.climate.copernicus.eu/datasets/projections-cordex-domains-single-levels?tab=download  , https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.15454/BPN57S , https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_RPG
cultures observées https://www.data.gouv.fr/datasets/rpg?resource_id=9209a9b3-f2f1-4be7-adf8-9d2665f4a1c5 
rendements historiques ; https://ecocrop.apps.fao.org/ecocrop 
Le schéma SQL est disponible dans le dépôt, dans le dossier dédié aux scripts SQL.

Machine learning
Deux approches ont été étudiées.
Modèle à l’échelle des mailles
Le premier modèle travaille à une résolution géographique d’environ 12 km pour les trois cultures.
Variables utilisées :
température moyenne ;
précipitations cumulées ;
rayonnement solaire ;
type de sol ;
compatibilité entre le sol réel et les sols préférés de la culture.
Plusieurs modèles ont été comparés :
régression linéaire ;
Ridge ;
Lasso ;
Random Forest ;
XGBoost.
Les années 2023 et 2024 ont été conservées comme données de test non vues pendant l’entraînement.
Résultats observés :
Culture
R² approximatif
Vigne
0,08
Pois chiche
0,17
Pomme de terre
0,28

Ces performances faibles montrent que les variables disponibles n’expliquent qu’une partie des variations de rendement.
Modèle départemental consacré à la vigne
Une seconde expérimentation a été menée à l’échelle départementale :
744 observations ;
61 départements ;
période 2010–2024 ;
entraînement sur 2010–2020 ;
test sur 2021–2024.
Résultats observés :
Modèle
R² approximatif
Baseline fondée sur la moyenne historique départementale
0,47
Ridge
0,49
XGBoost
0,38

Le modèle Ridge obtient de bonnes performances sur les départements déjà observés, mais dépend fortement de leur historique. Lorsqu’il est testé sur un département inconnu, sa capacité de généralisation se dégrade fortement.
XGBoost a été retenu pour certaines projections lointaines, car ses résultats restent plus stables et évitent notamment la génération de rendements négatifs.
Interprétation
Le rendement dépend également de variables qui ne sont pas disponibles dans les données actuelles :
irrigation ;
pratiques agricoles ;
variétés cultivées ;
fertilisation ;
maladies ;
traitements ;
événements climatiques extrêmes ;
qualité et exhaustivité des données sur les sols.
Les projections doivent donc être considérées comme des estimations exploratoires, et non comme des prévisions certaines.

Sources de données
Le projet exploite plusieurs sources publiques ou institutionnelles, notamment :
Copernicus / CORDEX pour les projections climatiques ;
IGN pour certaines données géographiques ;
données publiques agricoles et de rendement ;
données pédologiques ;
données climatiques historiques ;
données stockées et préparées dans AWS S3.
Les jeux de données peuvent être :
mesurés ;
calculés ;
simulés ;
agrégés à des granularités différentes.
Cette hétérogénéité constitue l’un des principaux défis méthodologiques du projet.
Les liens exacts, licences et conditions d’utilisation de chaque jeu de données doivent être documentés dans un fichier dédié, par exemple docs/data-sources.md.

Technologies
Langages et frameworks
Python
SQL
Streamlit
FastAPI
PostgreSQL
Data et machine learning
Pandas
NumPy
scikit-learn
XGBoost
notebooks Jupyter
bibliothèques géospatiales Python
Infrastructure
AWS S3
Neon PostgreSQL
Docker
Docker Compose
Visualisation
Streamlit
cartes interactives
graphiques et indicateurs métiers

Installation
Prérequis
Git
Docker Desktop
Docker Compose
accès aux services externes utilisés par le projet :
Neon ;
AWS S3 ;
Copernicus ;
API FastAPI, selon le mode d’exécution.
Cloner le dépôt
git clone <URL_DU_DEPOT>
cd Harvest-Games

Configuration
Créez les fichiers de configuration attendus par le projet avant de lancer l’application.
Exemple de variables d’environnement
Les noms ci-dessous doivent être adaptés aux noms réellement utilisés dans le code.
NEON_DATABASE_URL=postgresql://...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
S3_BUCKET_NAME=...
COPERNICUS_API_KEY=...
FASTAPI_URL=http://api:8000
Secrets Streamlit
Exemple de fichier .streamlit/secrets.toml :
NEON_DATABASE_URL = "postgresql://..."

AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_DEFAULT_REGION = "..."
S3_BUCKET_NAME = "..."

COPERNICUS_API_KEY = "..."
FASTAPI_URL = "http://api:8000"
[!IMPORTANT]
Les secrets, clés d’API et identifiants ne doivent jamais être versionnés dans Git. Vérifiez que les fichiers .env et .streamlit/secrets.toml figurent bien dans .gitignore.

Lancement
La commande officielle pour lancer le projet est :
docker compose up --build
Une fois les conteneurs démarrés, ouvrez dans votre navigateur l’URL indiquée dans les logs Docker.
Le point d’entrée principal de l’application Streamlit est :
main.py
Pour arrêter les services :
docker compose down

Structure du dépôt
La structure exacte peut être adaptée à l’organisation finale du projet.
Harvest-Games/
├── main.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── api/
│   ├── main.py
│   └── ...
├── etl/
│   ├── extract/
│   ├── transform/
│   ├── load/
│   └── ...
├── notebooks/
│   ├── vigne_ml.ipynb
│   └── ...
├── models/
│   └── ...
├── sql/
│   ├── schema.sql
│   └── ...
├── data/
│   └── ...
├── docs/
│   ├── screenshots/
│   ├── demo/
│   └── data-sources.md
├── tests/
│   └── ...
└── README.md

Résultats et limites
Résultats
Harvest Games permet de :
réunir plusieurs familles de données dans une architecture cohérente ;
visualiser les effets possibles du changement climatique ;
comparer plusieurs cultures ;
identifier des zones maintenues, perdues ou nouvellement favorables ;
construire des modèles exploratoires de rendement ;
proposer une interface pédagogique accessible à des acteurs non techniques.
Limites actuelles
Granularité des données
Les jeux de données n’ont pas tous la même résolution :
parcelle ;
maille climatique ;
département ;
région.
Le projet utilise actuellement des mailles d’environ 12 km. Une version ultérieure pourra viser une résolution plus fine.
Rendements publics
Les données publiques de rendement sont principalement disponibles à l’échelle départementale, ce qui limite la précision d’un modèle cartographique plus fin.
Qualité des données pédologiques
Une partie importante des mailles ne possède pas d’information complète sur les sols.
Variables manquantes
Le rendement dépend de nombreuses variables non intégrées :
irrigation ;
pratiques culturales ;
variétés ;
maladies ;
traitements ;
fertilisation ;
accidents climatiques ;
accès à l’eau.
Biais climatiques
Les projections CORDEX utilisées présentent un biais systématique identifié dans le cadre du projet. Une correction d’environ 0,9 °C doit être étudiée et validée avant toute utilisation avancée.
Généralisation des modèles
Certains modèles apprennent principalement les différences historiques entre territoires et généralisent mal à des départements jamais observés.

Perspectives
Amélioration de la précision en France
Analyse pluriannuelle
Élargissement des cultures
Comparaison multiculture
Passage à une résolution plus fine
Extension européenne et PAC

Équipe
Antoine
Architecture du projet, cohérence technique et intégration.
GitHub : @hantoinen
Céline Maussang
Expérience utilisateur, design de l’application, visualisations et interface Streamlit. https://harvestgames-eedugfpxudqyajst6fcial.streamlit.app/ 
GitHub : @maussangceline-ops
Ugo
Machine learning, entraînement, comparaison et évaluation des modèles.
GitHub : @Smashoow

Statut du projet
Projet étudiant terminé pour présentation initiale. Il sera repris pour être poussé dans des développements futurs.
Le dépôt reste une preuve de concept pouvant servir de base à une version plus précise, plus complète et déployable à plus grande échelle.

Licence
Aucune licence n’est actuellement associée à ce dépôt.
En l’absence de licence explicite, le code reste protégé par le droit d’auteur et ne peut pas être librement réutilisé, modifié ou redistribué sans autorisation des auteurs.

