# Dictionnaire de données - HarvestGames

## Schéma : `public`

### 1. Table : `dim_constraints`
* **Description :** Contient les contraintes agronomiques, climatiques et pédologiques requises ou optimales pour chaque culture. Ces données sont issues de sites agronomiques pour établir les zones "idéales" à la production d'une culture.

| Champ | Type | Nullable | Description & Règles |
| :--- | :--- | :--- | :--- |
| `culture_name` | `text` | Oui | Nom usuel de la culture (pomme de terre, pois chiche, vigne à raisin de cuve) |
| `temp_opt_min` | `bigint` | Oui | Température optimale minimale ($^\circ\text{C}$). Température au dessous de laquelle la croissance de la culture stagne et/ou s'interrompt par la mort de la plante. Elle n'est plus considérée comme productive |
| `temp_opt_max` | `bigint` | Oui | Température optimale maximale ($^\circ\text{C}$). Température au dessus de laquelle la croissance de la culture stagne et/ou s'interrompt par la mort de la plante. Elle n'est plus considérée comme productive |
| `temp_abs_min` | `bigint` | Oui | Température absolue minimale tolérée ($^\circ\text{C}$). Balise basse de la température idéale de croissance |
| `temp_abs_max` | `bigint` | Oui | Température absolue maximale tolérée ($^\circ\text{C}$). Balise haute de la température idéale de croissance |
| `pluie_optimale_min_mm_an` | `bigint` | Oui | Précipitations annuelles optimales minimales (mm/an). Balise basse des précipitations (puie) nécessaire à la croissance rentable d'une culture |
| `pluie_optimale_max_mm_an` | `bigint` | Oui | Précipitations annuelles optimales maximales (mm/an). Balise hautes des précipitations (puie) nécessaire à la croissance rentable d'une culture. Attention les cultures peuvent souffrir de trop d'eau |
| `ph_optimal_min` | `double precision` | Oui | pH du sol optimal minimal. |
| `ph_optimal_max` | `double precision` | Oui | pH du sol optimal maximal. |
| `required_soil_type` | `text` | Oui | Type de sol requis. (voir dictionnaire des sols plus bas) |
| `temp_saison_min` | `double precision` | Oui | Température minimale de saison. |
| `temp_saison_max` | `double precision` | Oui | Température maximale de saison. |
| `precip_saison_min` | `double precision` | Oui | Précipitations minimales de saison. |
| `precip_saison_max` | `double precision` | Oui | Précipitations maximales de saison. |
| `rayonnement_saison_min_kwh_m2` | `double precision` | Oui | Rayonnement solaire minimal de saison ($\text{kWh/m}^2$). |
| `rayonnement_saison_max_kwh_m2` | `double precision` | Oui | Rayonnement solaire maximal de saison ($\text{kWh/m}^2$). |
| `ensoleillement_saison_min_pct` | `double precision` | Oui | Ensoleillement minimal de saison (%). |
| `ensoleillement_saison_max_pct` | `double precision` | Oui | Ensoleillement maximal de saison (%). |
| `profondeur_sol_preferee_cm` | `text` | Oui | Profondeur de sol préférée (cm). Racines |
| `texture_sol_preferee_type` | `text` | Oui | Type de texture de sol préféré. voir soil_type dans les listes en bas de document|
| `texture_sol_preferee_drainage` | `text` | Oui | Drainage du sol préféré. |
| `texture_sol_frequente_classe` | `text` | Oui | Classe de texture de sol fréquente. |
| `texture_sol_frequente_detail` | `text` | Oui | Détail de la texture de sol fréquente. |
| `preferred_soil_family_1` | `text` | Oui | Famille de sol préférée (1er choix majoritaire sur les cultures 2024). |
| `preferred_soil_family_2` | `text` | Oui | Famille de sol préférée (2e choix sur les cultures 2024)). |
| `preferred_soil_family_3` | `text` | Oui | Famille de sol préférée (3e choix sur les cultures 2024)). |
| `culture_id` | `bigint` | Oui | Identifiant unique de la culture (Clé de liaison). |

---

### 2. Table : `dim_coordinates` - source IGN
* **Description :** Référentiel géographique des coordonnées spatiales (latitude / longitude).

| Champ | Type | Nullable | Description & Règles |
| :--- | :--- | :--- | :--- |
| `lat` | `double precision` | Oui | Latitude géographique. |
| `lon` | `double precision` | Oui | Longitude géographique. |
| `coordinates_id` | `bigint` | Oui | Identifiant unique des coordonnées (Clé primaire). |

---

### 3. Table : `dim_soils` - source INRAE
* **Description :** Référentiel pédologique et administratif (sols, départements, régions).

| Champ | Type | Nullable | Description & Règles |
| :--- | :--- | :--- | :--- |
| `soil_family_type` | `text` | Oui | Type de famille de sol. |
| `soil_type` | `text` | Oui | Type de sol détaillé. |
| `department_code` | `text` | Oui | Code officiel du département. |
| `department_name` | `text` | Oui | Nom du département. |
| `region_code` | `text` | Oui | Code officiel de la région. |
| `region_name` | `text` | Oui | Nom de la région. |
| `coordinates_id` | `bigint` | Oui | Identifiant des coordonnées géographiques (Clé étrangère). |

---

### 4. Table : `src_climate_future` - source COPERNICUS
* **Description :** Données climatiques prévisionnelles/futures associées aux coordonnées.

| Champ | Type | Nullable | Description & Règles |
| :--- | :--- | :--- | :--- |
| `coordinates_id` | `integer` | Oui | Identifiant des coordonnées géographiques (Clé étrangère). |
| `date` | `date` | Oui | Date de la mesure/prévision. |
| `year` | `smallint` | Oui | Année. |
| `month` | `smallint` | Oui | Mois. |
| `temp_mean_future` | `real` | Oui | Température moyenne future prévue ($^\circ\text{C}$). |
| `precip_future` | `real` | Oui | Précipitations futures prévues (mm). |
| `solar_rad_future` | `real` | Oui | Rayonnement solaire futur prévu. |

---

### 5. Table : `src_climate_past` - source AGRA5
* **Description :** Données climatiques historiques observées.

| Champ | Type | Nullable | Description & Règles |
| :--- | :--- | :--- | :--- |
| `coordinates_id` | `integer` | Oui | Identifiant des coordonnées géographiques (Clé étrangère). |
| `date` | `timestamp` | Oui | Horodatage de la mesure passée. |
| `year` | `smallint` | Oui | Année. |
| `month` | `smallint` | Oui | Mois. |
| `temp_mean_past` | `real` | Oui | Température moyenne historique ($^\circ\text{C}$). |
| `precip_past` | `real` | Oui | Précipitations historiques (mm). |
| `solar_rad_past` | `real` | Oui | Rayonnement solaire historique. |

---

### 6. Table : `src_cultures_2024` - Source AGRESTE / INRAE
* **Description :** Données de cultures pour l'année 2024 (parcelles, surfaces).

| Champ | Type | Nullable | Description & Règles |
| :--- | :--- | :--- | :--- |
| `time` | `date` | Oui | Date ou période de relevé pour 2024. |
| `temp_celsius` | `double precision` | Oui | Température mesurée ($^\circ\text{C}$). |
| `nb_parcelles_5km` | `bigint` | Oui | Nombre de parcelles dans un rayon de 5 km. |
| `surface_ha_5km` | `double precision` | Oui | Surface totale en hectares dans un rayon de 5 km. |
| `culture_id` | `bigint` | Oui | Identifiant de la culture (Clé étrangère). |
| `coordinates_id` | `bigint` | Oui | Identifiant des coordonnées (Clé étrangère). |

---

### 7. Table : `src_yields_2010_2025` - Source ECOCROP (2025 incomplète)
* **Description :** Historique des rendements et productions agricoles par département entre 2010 et 2025.

| Champ | Type | Nullable | Description & Règles |
| :--- | :--- | :--- | :--- |
| `code_departement` | `text` | Oui | Code du département. |
| `nom_departement` | `text` | Oui | Nom du département. |
| `culture_id` | `bigint` | Oui | Identifiant de la culture (Clé étrangère). |
| `annee` | `bigint` | Oui | Année de la récolte (entre 2010 et 2025). |
| `rendement_q_ha` | `double precision` | Oui | Rendement en quintaux par hectare ($\text{q/ha}$). |
| `surface_ha` | `double precision` | Oui | Surface cultivée en hectares ($\text{ha}$). |
| `production_q` | `double precision` | Oui | Production totale en quintaux ($\text{q}$). |
| `statut_qualite` | `bigint` | Oui | Indicateur ou statut de qualité de la donnée. |
| `a_historique_culture` | `boolean` | Oui | Indique si la culture possède un historique (`TRUE`/`FALSE`). |


# Listes
soil_family_type = {
    'B': 'Cambisols (sols bruns modérément évolués)',
    'E': 'Rendzines / Leptosols calcaro-magnésiens',
    'L': 'Luvisols (sols lessivés / argileux)',
    'J': 'Fluvisols (sols d\'alluvions / vallées)',
    'I': 'Lithosols (sols très minces sur roche dure)',
    'R': 'Régosols (sols peu évolués sur matériau meuble)',
    'P': 'Podzols (inclut podzols dégradés)',
    'D': 'Podzols (inclut podzols dégradés)',
    'Q': 'Arénosols (sols très sableux)',
    'U': 'Rankers (sols silatés / acides)',
    'Z': 'Solonchaks (sols à contrainte saline)',
    'H': 'Histosols (sols organiques / tourbières)'
}

soil_type = {
    1: 'Sableuse (Faible rétention)',
    2: 'Limoneuse (Rétention moyenne)',
    3: 'Argileuse (Forte rétention)',
    4: 'Très lourde / Argile lourde',
    5: 'Organique / Tourbeuse'
}
