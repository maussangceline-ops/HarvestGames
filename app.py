import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from sqlalchemy import text
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(layout="wide", page_title="Harvest Games")

# 1. Connexion à la base de données Neon
conn = st.connection(
    "postgresql", 
    type="sql", 
    url=st.secrets["NEON_DATABASE_URL"],
    pool_recycle=300,
    connect_args={"keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 5}
)

# 2. Fonctions de chargement des données
@st.cache_data
def load_constraints():
    return conn.query('SELECT * FROM dim_constraints;', ttl=600)

@st.cache_data
def load_current_culture_data(culture_id, annee_cible=2054):
    query = """
        WITH cultures_par_maille AS (
            SELECT
                coordinates_id,
                COUNT(DISTINCT parcel_id) AS nb_parcelles,
                SUM(surf_ha) AS surface_ha_exacte
            FROM src_cultures_plots_2024
            WHERE culture_id = :culture_id
              AND surf_ha > 0
            GROUP BY coordinates_id
        ),

        climat_2024 AS (
            SELECT
                coordinates_id,
                EXTRACT(MONTH FROM "date")::integer AS month,
                AVG(temp_mean_past) AS temp_2024
            FROM src_climate_past
            WHERE "date" BETWEEN DATE '2024-04-01' AND DATE '2024-08-01'
            GROUP BY
                coordinates_id,
                EXTRACT(MONTH FROM "date")
        ),

        climat_2054 AS (
            SELECT
                coordinates_id,
                month,
                AVG(temp_mean_future) AS temp_2054,
                AVG(precip_future) AS precip_future,
                AVG(solar_rad_future) AS solar_rad_future
            FROM src_climate_future
            WHERE year = :annee_cible
              AND month BETWEEN 4 AND 8
            GROUP BY
                coordinates_id,
                month
        )

        SELECT
            cultures.coordinates_id,
            coord.lat,
            coord.lon,
            futur.month,
            passe.temp_2024,
            futur.temp_2054,
            futur.precip_future,
            futur.solar_rad_future,
            cultures.nb_parcelles,
            cultures.surface_ha_exacte

        FROM cultures_par_maille cultures

        JOIN dim_coordinates coord
            ON cultures.coordinates_id = coord.coordinates_id

        JOIN climat_2054 futur
            ON cultures.coordinates_id = futur.coordinates_id

        LEFT JOIN climat_2024 passe
            ON cultures.coordinates_id = passe.coordinates_id
            AND futur.month = passe.month;
    """

    return conn.query(
        query,
        params={
            "culture_id": int(culture_id),
            "annee_cible": int(annee_cible)
        },
        ttl=0
    )

@st.cache_data
def load_national_stats(culture_id):
    query = """
        SELECT
            COUNT(DISTINCT parcel_id) AS nb_parcelles,
            COUNT(DISTINCT coordinates_id) AS nb_mailles,
            SUM(surf_ha) AS total_ha
        FROM src_cultures_plots_2024
        WHERE culture_id = :culture_id
          AND surf_ha > 0;
    """

    return conn.query(
        query,
        params={
            "culture_id": int(culture_id)
        },
        ttl=0
    )

@st.cache_data
def load_departments_geojson():
    with open(
        "assets/departements-1000m.geojson",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f) 

# DEBUT DU CODE PP
df_constraints = load_constraints()
liste_cultures = df_constraints['culture_name'].tolist() if not df_constraints.empty else ["Vigne à raisin de cuve"]

culture_info = {
    "Vigne à raisin de cuve": {"picto": "🍇", "id": 1},
    "Pomme de terre": {"picto": "🥔", "id": 3},
    "Pois chiche": {"picto": "🌱", "id": 2}
}

@st.cache_data
def load_yield_training_data(culture_id):
    """
    Une ligne par culture, maille et année.

    Les rendements proviennent de src_yields_cordex_2010_2025.
    Les variables climatiques proviennent de src_climate_past.
    La saison étudiée va d'avril à août.
    """

    query = """
        WITH climat_historique_mensuel AS (
            SELECT
                coordinates_id,
                year,
                month,

                AVG(temp_mean_past) AS temp_mois,
                AVG(precip_past) AS precip_mois,
                AVG(solar_rad_past) AS solar_mois

            FROM src_climate_past

            WHERE year BETWEEN 2010 AND 2025
              AND month BETWEEN 4 AND 8

            GROUP BY
                coordinates_id,
                year,
                month
        ),

        climat_historique_saison AS (
            SELECT
                coordinates_id,
                year,

                AVG(temp_mois) AS temp_saison,
                MAX(temp_mois) AS temp_max_saison,
                SUM(precip_mois) AS precip_cumul,
                AVG(solar_mois) AS solar_moy,

                SUM(
                    solar_mois
                    * 24
                    * EXTRACT(
                        DAY FROM (
                            MAKE_DATE(year, month, 1)
                            + INTERVAL '1 month'
                            - INTERVAL '1 day'
                        )
                    )
                    / 1000.0
                ) AS rayonnement_saison_kwh_m2,

                COUNT(*) AS nb_mois

            FROM climat_historique_mensuel

            GROUP BY
                coordinates_id,
                year
        ),

        sols AS (
            SELECT
                coordinates_id,

                MODE() WITHIN GROUP (
                    ORDER BY soil_family_type
                ) AS soil_family_type

            FROM dim_soils

            GROUP BY coordinates_id
        )

        SELECT
            rendements.coordinates_id,
            rendements.culture_id,
            rendements.year,
            rendements.rendement_q_ha,
            rendements.surface_ha_12_5km,

            climat.temp_saison,
            climat.temp_max_saison,
            climat.precip_cumul,
            climat.solar_moy,
            climat.rayonnement_saison_kwh_m2,

            COALESCE(
                sols.soil_family_type,
                'inconnu'
            ) AS soil_family_type

        FROM src_yields_cordex_2010_2025 rendements

        JOIN climat_historique_saison climat
            ON rendements.coordinates_id = climat.coordinates_id
            AND rendements.year = climat.year

        LEFT JOIN sols
            ON rendements.coordinates_id = sols.coordinates_id

        WHERE rendements.culture_id = :culture_id

          AND rendements.rendement_q_ha IS NOT NULL
          AND rendements.rendement_q_ha > 0

          AND COALESCE(
              rendements.surface_ha_12_5km,
              0
          ) > 0

          AND climat.nb_mois = 5;
    """

    return conn.query(
        query,
        params={
            "culture_id": int(culture_id)
        },
        ttl=0
    )

@st.cache_data
def load_future_yield_features(annee_sel):
    """
    Prépare les mêmes variables que celles utilisées pour entraîner
    le modèle, mais avec le climat futur.
    """

    query = """
        WITH climat_futur_mensuel AS (
            SELECT
                coordinates_id,
                year,
                month,

                AVG(temp_mean_future) AS temp_mois,
                AVG(precip_future) AS precip_mois,
                AVG(solar_rad_future) AS solar_mois

            FROM src_climate_future

            WHERE year = :annee_sel
              AND month BETWEEN 4 AND 8

            GROUP BY
                coordinates_id,
                year,
                month
        ),

        climat_futur_saison AS (
            SELECT
                coordinates_id,
                year,

                AVG(temp_mois) AS temp_saison,
                MAX(temp_mois) AS temp_max_saison,
                SUM(precip_mois) AS precip_cumul,
                AVG(solar_mois) AS solar_moy,

                SUM(
                    solar_mois
                    * 24
                    * EXTRACT(
                        DAY FROM (
                            MAKE_DATE(year, month, 1)
                            + INTERVAL '1 month'
                            - INTERVAL '1 day'
                        )
                    )
                    / 1000.0
                ) AS rayonnement_saison_kwh_m2,

                COUNT(*) AS nb_mois

            FROM climat_futur_mensuel

            GROUP BY
                coordinates_id,
                year
        ),

        sols AS (
            SELECT
                coordinates_id,

                MODE() WITHIN GROUP (
                    ORDER BY soil_family_type
                ) AS soil_family_type

            FROM dim_soils

            GROUP BY coordinates_id
        )

        SELECT
            climat.coordinates_id,
            climat.year,

            coord.lat,
            coord.lon,

            climat.temp_saison,
            climat.temp_max_saison,
            climat.precip_cumul,
            climat.solar_moy,
            climat.rayonnement_saison_kwh_m2,

            COALESCE(
                sols.soil_family_type,
                'inconnu'
            ) AS soil_family_type

        FROM climat_futur_saison climat

        JOIN dim_coordinates coord
            ON climat.coordinates_id = coord.coordinates_id

        LEFT JOIN sols
            ON climat.coordinates_id = sols.coordinates_id

        WHERE climat.nb_mois = 5;
    """

    return conn.query(
        query,
        params={
            "annee_sel": int(annee_sel)
        },
        ttl=0
    )

FEATURES_RENDEMENT_NUMERIQUES = [
    "temp_saison",
    "temp_max_saison",
    "precip_cumul",
    "solar_moy"
]

FEATURES_RENDEMENT_CATEGORIELLES = [
    "soil_family_type"
]

POIDS_HARVEST_SCORE = {
    "rendement": 0.40,
    "temperature": 0.25,
    "precipitations": 0.15,
    "rayonnement": 0.10,
    "sol": 0.10
}

@st.cache_resource
def train_yield_model(culture_id):

    df_historique = load_yield_training_data(
        culture_id=culture_id
    ).copy()

    colonnes_numeriques = FEATURES_RENDEMENT_NUMERIQUES
    colonnes_categorielles = FEATURES_RENDEMENT_CATEGORIELLES

    cible = "rendement_q_ha"

    colonnes_utiles = (
        colonnes_numeriques
        + colonnes_categorielles
        + [
            cible,
            "year",
            "coordinates_id",
            "rayonnement_saison_kwh_m2"
        ]
    )

    df_model = df_historique[
        colonnes_utiles
    ].copy()

    df_model[cible] = pd.to_numeric(
        df_model[cible],
        errors="coerce"
    )

    df_model = df_model.dropna(
        subset=[cible]
    )

    if len(df_model) < 50:
        raise ValueError(
            "Le nombre de données historiques est insuffisant "
            "pour entraîner un modèle de rendement fiable."
        )

    # Suppression prudente des rendements extrêmement atypiques.
    rendement_min = df_model[cible].quantile(0.01)
    rendement_max = df_model[cible].quantile(0.99)

    df_model = df_model[
        df_model[cible].between(
            rendement_min,
            rendement_max
        )
    ].copy()

    preprocesseur = ColumnTransformer(
        transformers=[
            (
                "variables_numeriques",
                Pipeline(
                    steps=[
                        (
                            "imputation",
                            SimpleImputer(
                                strategy="median"
                            )
                        )
                    ]
                ),
                colonnes_numeriques
            ),
            (
                "variables_categorielles",
                Pipeline(
                    steps=[
                        (
                            "imputation",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "encodage",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )
                    ]
                ),
                colonnes_categorielles
            )
        ]
    )

    modele = Pipeline(
        steps=[
            (
                "preparation",
                preprocesseur
            ),
            (
                "regression",
                RandomForestRegressor(
                    n_estimators=40,
                    max_depth=10,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    variables_modele = (
        colonnes_numeriques
        + colonnes_categorielles
    )

    # ----------------------------------
    # Validation temporelle
    # ----------------------------------

    df_entrainement = df_model[
        df_model["year"] <= 2022
    ]

    df_validation = df_model[
        df_model["year"] >= 2023
    ]

    metriques_modele = {
        "mae": None,
        "r2": None,
        "nb_lignes": len(df_model)
    }

    if (
        len(df_entrainement) >= 50
        and len(df_validation) >= 10
    ):
        modele.fit(
            df_entrainement[variables_modele],
            df_entrainement[cible]
        )

        predictions_validation = modele.predict(
            df_validation[variables_modele]
        )

        metriques_modele["mae"] = mean_absolute_error(
            df_validation[cible],
            predictions_validation
        )

        metriques_modele["r2"] = r2_score(
            df_validation[cible],
            predictions_validation
        )

    # Réentraînement final sur toute la période 2010–2025.
    modele.fit(
        df_model[variables_modele],
        df_model[cible]
    )

    # Le 90e percentile historique devient le rendement
    # correspondant à un score rendement de 100.
    rendement_reference_haut = (
        df_model[cible].quantile(0.90)
    )

    reference_par_maille = (
        df_model
        .groupby(
            "coordinates_id",
            as_index=False
        )
        .agg(
            rendement_reference=(
                "rendement_q_ha",
                "mean"
            ),
            temp_saison_reference=(
                "temp_saison",
                "mean"
            ),
            temp_max_reference=(
                "temp_max_saison",
                "mean"
            ),
            precip_cumul_reference=(
                "precip_cumul",
                "mean"
            ),
            solar_moy_reference=(
                "solar_moy",
                "mean"
            ),
            rayonnement_reference_kwh_m2=(
                "rayonnement_saison_kwh_m2",
                "mean"
            )
        )
    )

    valeurs_reference_culture = {
        "rendement_reference": (
            df_model["rendement_q_ha"].median()
        ),
        "temp_saison_reference": (
            df_model["temp_saison"].median()
        ),
        "temp_max_reference": (
            df_model["temp_max_saison"].median()
        ),
        "precip_cumul_reference": (
            df_model["precip_cumul"].median()
        ),
        "solar_moy_reference": (
            df_model["solar_moy"].median()
        ),
        "rayonnement_reference_kwh_m2": (
            df_model["rayonnement_saison_kwh_m2"].median()
        )
    }

    return (
        modele,
        rendement_reference_haut,
        reference_par_maille,
        valeurs_reference_culture,
        metriques_modele
    )

# ======================================
# CALCUL DES SOUS-SCORES DU HARVEST SCORE
# ======================================

def calculer_score_intervalle(
    valeur,
    borne_optimale_min,
    borne_optimale_max,
    limite_absolue_min=None,
    limite_absolue_max=None
):
    """
    Renvoie un score compris entre 0 et 100.

    Le score vaut 100 dans l'intervalle optimal, puis diminue
    progressivement jusqu'à 0 aux limites absolues.
    """

    valeurs_obligatoires = [
        valeur,
        borne_optimale_min,
        borne_optimale_max
    ]

    if any(pd.isna(element) for element in valeurs_obligatoires):
        return 0.0

    valeur = float(valeur)
    borne_optimale_min = float(borne_optimale_min)
    borne_optimale_max = float(borne_optimale_max)

    amplitude = max(
        borne_optimale_max - borne_optimale_min,
        0.0001
    )

    if limite_absolue_min is None or pd.isna(limite_absolue_min):
        limite_absolue_min = max(
            0.0,
            borne_optimale_min - amplitude
        )

    if limite_absolue_max is None or pd.isna(limite_absolue_max):
        limite_absolue_max = borne_optimale_max + amplitude

    limite_absolue_min = float(limite_absolue_min)
    limite_absolue_max = float(limite_absolue_max)

    if borne_optimale_min <= valeur <= borne_optimale_max:
        return 100.0

    if valeur < borne_optimale_min:
        if valeur <= limite_absolue_min:
            return 0.0

        denominateur = borne_optimale_min - limite_absolue_min
        if denominateur <= 0:
            return 0.0

        return float(np.clip(
            100
            * (valeur - limite_absolue_min)
            / denominateur,
            0,
            100
        ))

    if valeur >= limite_absolue_max:
        return 0.0

    denominateur = limite_absolue_max - borne_optimale_max
    if denominateur <= 0:
        return 0.0

    return float(np.clip(
        100
        * (limite_absolue_max - valeur)
        / denominateur,
        0,
        100
    ))

# ======================================
# CONSTRUCTION DU HARVEST SCORE
# ======================================

def build_harvest_score_data(
    culture_id,
    annee_sel,
    df_constraints,
    coefficient_nouvelles_zones=0.5
):
    """
    Prédit le rendement potentiel et calcule le Harvest Score par maille.

    Le résultat est relié aux statuts de la Carte 2 :
    - verte : surface historique maintenue ;
    - rouge : surface historique perdue, contribution future nulle ;
    - bleue : nouvelle surface potentielle estimée avec un coefficient prudent.
    """

    (
        modele,
        rendement_reference_haut,
        reference_par_maille,
        valeurs_reference_culture,
        metriques_modele
    ) = train_yield_model(
        culture_id=culture_id
    )

    df_futur = load_future_yield_features(
        annee_sel=annee_sel
    ).copy()

    if df_futur.empty:
        return df_futur, metriques_modele

    # Récupération des statuts vert, rouge et bleu de la Carte 2.
    df_projection_score = load_crop_projection(
        annee_sel=annee_sel,
        culture_id=culture_id
    ).copy()

    if df_projection_score.empty:
        return pd.DataFrame(), metriques_modele

    colonnes_projection = [
        "coordinates_id",
        "department_code",
        "department_name",
        "statut",
        "historique_2024",
        "surface_ha_12_5km",
        "raison_echec"
    ]

    df_projection_score = (
        df_projection_score[colonnes_projection]
        .drop_duplicates(
            subset=["coordinates_id"],
            keep="first"
        )
    )

    # La jointure interne conserve uniquement les mailles historiques
    # ou futures aptes identifiées par la Carte 2.
    df_futur = df_futur.merge(
        df_projection_score,
        on="coordinates_id",
        how="inner",
        validate="one_to_one"
    )

    if df_futur.empty:
        return df_futur, metriques_modele

    variables_modele = (
        FEATURES_RENDEMENT_NUMERIQUES
        + FEATURES_RENDEMENT_CATEGORIELLES
    )

    df_futur["rendement_potentiel_q_ha"] = modele.predict(
        df_futur[variables_modele]
    )

    df_futur["rendement_potentiel_q_ha"] = (
        df_futur["rendement_potentiel_q_ha"]
        .clip(lower=0)
    )

    df_futur = df_futur.merge(
        reference_par_maille,
        on="coordinates_id",
        how="left"
    )

    df_futur["reference_locale_disponible"] = (
        df_futur["rendement_reference"].notna()
    )

    for colonne, valeur_defaut in valeurs_reference_culture.items():
        df_futur[colonne] = df_futur[colonne].fillna(
            valeur_defaut
        )

    df_futur["type_reference"] = np.where(
        df_futur["reference_locale_disponible"],
        "Historique local 2010-2025",
        "Médiane de la culture 2010-2025"
    )

    lignes_contrainte = df_constraints.loc[
        df_constraints["culture_id"] == culture_id
    ]

    if lignes_contrainte.empty:
        raise ValueError(
            f"Aucune contrainte trouvée pour la culture {culture_id}."
        )

    contrainte = lignes_contrainte.iloc[0]

    sols_preferes = {
        str(sol).strip().lower()
        for sol in [
            contrainte["preferred_soil_family_1"],
            contrainte["preferred_soil_family_2"],
            contrainte["preferred_soil_family_3"]
        ]
        if pd.notna(sol)
    }

    df_futur["score_sol"] = np.where(
        df_futur["soil_family_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(sols_preferes),
        100.0,
        0.0
    )

    # Rendement : le 90e percentile historique vaut 100.
    rendement_reference_haut = max(
        float(rendement_reference_haut),
        0.0001
    )

    df_futur["score_rendement_reference"] = (
        df_futur["rendement_reference"]
        / rendement_reference_haut
        * 100
    ).clip(0, 100)

    df_futur["score_rendement_futur"] = (
        df_futur["rendement_potentiel_q_ha"]
        / rendement_reference_haut
        * 100
    ).clip(0, 100)

    # Température.
    temp_min = (
        contrainte["temp_saison_min"]
        if pd.notna(contrainte["temp_saison_min"])
        else contrainte["temp_opt_min"]
    )
    temp_max = (
        contrainte["temp_saison_max"]
        if pd.notna(contrainte["temp_saison_max"])
        else contrainte["temp_opt_max"]
    )

    df_futur["score_temperature_reference"] = (
        df_futur["temp_saison_reference"].apply(
            lambda valeur: calculer_score_intervalle(
                valeur,
                temp_min,
                temp_max,
                contrainte["temp_abs_min"],
                contrainte["temp_abs_max"]
            )
        )
    )

    df_futur["score_temperature_futur"] = (
        df_futur["temp_saison"].apply(
            lambda valeur: calculer_score_intervalle(
                valeur,
                temp_min,
                temp_max,
                contrainte["temp_abs_min"],
                contrainte["temp_abs_max"]
            )
        )
    )

    # Précipitations.
    precip_min = float(contrainte["precip_saison_min"])
    precip_max = float(contrainte["precip_saison_max"])
    amplitude_precip = max(
        precip_max - precip_min,
        0.0001
    )
    limite_precip_max = precip_max + amplitude_precip

    df_futur["score_precip_reference"] = (
        df_futur["precip_cumul_reference"].apply(
            lambda valeur: calculer_score_intervalle(
                valeur,
                precip_min,
                precip_max,
                0,
                limite_precip_max
            )
        )
    )

    df_futur["score_precip_futur"] = (
        df_futur["precip_cumul"].apply(
            lambda valeur: calculer_score_intervalle(
                valeur,
                precip_min,
                precip_max,
                0,
                limite_precip_max
            )
        )
    )

    # Rayonnement : même conversion que dans la Carte 2.
    ray_min = (
        float(contrainte["rayonnement_saison_min_kwh_m2"])
        / 3.6
    )
    ray_max = (
        float(contrainte["rayonnement_saison_max_kwh_m2"])
        / 3.6
    )
    amplitude_ray = max(
        ray_max - ray_min,
        0.0001
    )

    df_futur["score_rayonnement_reference"] = (
        df_futur["rayonnement_reference_kwh_m2"].apply(
            lambda valeur: calculer_score_intervalle(
                valeur,
                ray_min,
                ray_max,
                max(0, ray_min - amplitude_ray),
                ray_max + amplitude_ray
            )
        )
    )

    df_futur["score_rayonnement_futur"] = (
        df_futur["rayonnement_saison_kwh_m2"].apply(
            lambda valeur: calculer_score_intervalle(
                valeur,
                ray_min,
                ray_max,
                max(0, ray_min - amplitude_ray),
                ray_max + amplitude_ray
            )
        )
    )

    df_futur["harvest_score_reference"] = (
        df_futur["score_rendement_reference"]
        * POIDS_HARVEST_SCORE["rendement"]
        + df_futur["score_temperature_reference"]
        * POIDS_HARVEST_SCORE["temperature"]
        + df_futur["score_precip_reference"]
        * POIDS_HARVEST_SCORE["precipitations"]
        + df_futur["score_rayonnement_reference"]
        * POIDS_HARVEST_SCORE["rayonnement"]
        + df_futur["score_sol"]
        * POIDS_HARVEST_SCORE["sol"]
    ).clip(0, 100).round(1)

    df_futur["harvest_score_futur"] = (
        df_futur["score_rendement_futur"]
        * POIDS_HARVEST_SCORE["rendement"]
        + df_futur["score_temperature_futur"]
        * POIDS_HARVEST_SCORE["temperature"]
        + df_futur["score_precip_futur"]
        * POIDS_HARVEST_SCORE["precipitations"]
        + df_futur["score_rayonnement_futur"]
        * POIDS_HARVEST_SCORE["rayonnement"]
        + df_futur["score_sol"]
        * POIDS_HARVEST_SCORE["sol"]
    ).clip(0, 100).round(1)

    df_futur["evolution_score_points"] = (
        df_futur["harvest_score_futur"]
        - df_futur["harvest_score_reference"]
    ).round(1)

    df_futur["evolution_rendement_pct"] = (
        (
            df_futur["rendement_potentiel_q_ha"]
            - df_futur["rendement_reference"]
        )
        .div(
            df_futur["rendement_reference"].replace(
                0,
                pd.NA
            )
        )
        * 100
    ).round(1)

    # -------------------------------------------------
    # Estimation de la surface future par maille
    # -------------------------------------------------

    df_futur["surface_ha_12_5km"] = pd.to_numeric(
        df_futur["surface_ha_12_5km"],
        errors="coerce"
    ).fillna(0)

    df_futur["department_code"] = (
        df_futur["department_code"]
        .astype("string")
        .str.strip()
    )

    df_futur["department_name"] = (
        df_futur["department_name"]
        .astype("string")
        .str.strip()
    )

    surfaces_historiques = df_futur.loc[
        df_futur["historique_2024"]
        & (df_futur["surface_ha_12_5km"] > 0),
        "surface_ha_12_5km"
    ]

    surface_mediane_nationale = (
        float(surfaces_historiques.median())
        if not surfaces_historiques.empty
        else 0.0
    )

    medianes_departements = (
        df_futur.loc[
            df_futur["historique_2024"]
            & (df_futur["surface_ha_12_5km"] > 0)
            & df_futur["department_code"].notna()
            & df_futur["department_name"].notna()
        ]
        .groupby(
            ["department_code", "department_name"],
            as_index=False
        )
        .agg(
            surface_mediane_historique_maille=(
                "surface_ha_12_5km",
                "median"
            )
        )
    )

    df_futur = df_futur.merge(
        medianes_departements,
        on=["department_code", "department_name"],
        how="left"
    )

    df_futur[
        "surface_mediane_historique_maille"
    ] = (
        df_futur[
            "surface_mediane_historique_maille"
        ]
        .fillna(surface_mediane_nationale)
    )

    df_futur["surface_future_estimee_ha"] = np.select(
        [
            df_futur["statut"].eq("verte"),
            df_futur["statut"].eq("bleue"),
            df_futur["statut"].eq("rouge")
        ],
        [
            df_futur["surface_ha_12_5km"],
            (
                df_futur[
                    "surface_mediane_historique_maille"
                ]
                * float(coefficient_nouvelles_zones)
            ),
            0.0
        ],
        default=0.0
    )

    df_futur["surface_reference_2024_ha"] = np.where(
        df_futur["historique_2024"],
        df_futur["surface_ha_12_5km"],
        0.0
    )

    df_futur["surface_nouvelle_estimee_ha"] = np.where(
        df_futur["statut"].eq("bleue"),
        df_futur["surface_future_estimee_ha"],
        0.0
    )

    return df_futur, metriques_modele


def moyenne_ponderee(
    dataframe,
    colonne_valeur,
    colonne_poids
):
    """Calcule une moyenne pondérée en ignorant les valeurs invalides."""

    donnees = dataframe[
        [colonne_valeur, colonne_poids]
    ].copy()

    donnees[colonne_valeur] = pd.to_numeric(
        donnees[colonne_valeur],
        errors="coerce"
    )
    donnees[colonne_poids] = pd.to_numeric(
        donnees[colonne_poids],
        errors="coerce"
    )

    donnees = donnees.dropna()
    donnees = donnees[
        donnees[colonne_poids] > 0
    ]

    if donnees.empty:
        return np.nan

    return float(
        np.average(
            donnees[colonne_valeur],
            weights=donnees[colonne_poids]
        )
    )


def calculer_indicateurs_nationaux_harvest(
    df_harvest_score
):
    """
    Calcule les indicateurs nationaux.

    Les surfaces rouges restent dans la surface de référence mais ont
    une contribution future nulle. Elles diminuent donc directement
    le Harvest Score national.
    """

    if df_harvest_score.empty:
        return {}

    df = df_harvest_score.copy()

    surface_reference_2024 = float(
        df["surface_reference_2024_ha"].sum()
    )
    surface_maintenue = float(
        df.loc[
            df["statut"].eq("verte"),
            "surface_future_estimee_ha"
        ].sum()
    )
    surface_perdue = float(
        df.loc[
            df["statut"].eq("rouge"),
            "surface_reference_2024_ha"
        ].sum()
    )
    surface_nouvelle = float(
        df.loc[
            df["statut"].eq("bleue"),
            "surface_future_estimee_ha"
        ].sum()
    )
    surface_future = float(
        df["surface_future_estimee_ha"].sum()
    )

    # Le dénominateur comprend la surface 2024 et les nouvelles
    # surfaces estimées. Une surface rouge contribue donc avec 0 point.
    surface_score = (
        surface_reference_2024
        + surface_nouvelle
    )

    if surface_score > 0:
        numerateur_score = float(
            (
                df["harvest_score_futur"]
                * df["surface_future_estimee_ha"]
            ).sum()
        )
        harvest_score_national = (
            numerateur_score / surface_score
        )
    else:
        harvest_score_national = np.nan

    rendement_national_futur = moyenne_ponderee(
        df,
        "rendement_potentiel_q_ha",
        "surface_future_estimee_ha"
    )

    rendement_reference_national = moyenne_ponderee(
        df.loc[df["historique_2024"]],
        "rendement_reference",
        "surface_reference_2024_ha"
    )

    if (
        pd.notna(rendement_national_futur)
        and pd.notna(rendement_reference_national)
        and rendement_reference_national > 0
    ):
        evolution_rendement_pct = (
            (
                rendement_national_futur
                - rendement_reference_national
            )
            / rendement_reference_national
            * 100
        )
    else:
        evolution_rendement_pct = np.nan

    # Production potentielle = rendement moyen pondéré × surface productive.
    production_reference_2024_q = (
        rendement_reference_national * surface_reference_2024
        if pd.notna(rendement_reference_national)
        else np.nan
    )

    production_future_estimee_q = (
        rendement_national_futur * surface_future
        if pd.notna(rendement_national_futur)
        else np.nan
    )

    if (
        pd.notna(production_future_estimee_q)
        and pd.notna(production_reference_2024_q)
        and production_reference_2024_q > 0
    ):
        evolution_production_pct = (
            (
                production_future_estimee_q
                - production_reference_2024_q
            )
            / production_reference_2024_q
            * 100
        )
    else:
        evolution_production_pct = np.nan

    if surface_reference_2024 > 0:
        evolution_surface_pct = (
            (surface_future - surface_reference_2024)
            / surface_reference_2024
            * 100
        )
    else:
        evolution_surface_pct = np.nan

    return {
        "harvest_score_national": harvest_score_national,
        "rendement_national_futur": rendement_national_futur,
        "rendement_reference_national": rendement_reference_national,
        "evolution_rendement_pct": evolution_rendement_pct,
        "production_reference_2024_q": production_reference_2024_q,
        "production_future_estimee_q": production_future_estimee_q,
        "evolution_production_pct": evolution_production_pct,
        "surface_reference_2024_ha": surface_reference_2024,
        "surface_maintenue_ha": surface_maintenue,
        "surface_perdue_ha": surface_perdue,
        "surface_nouvelle_estimee_ha": surface_nouvelle,
        "surface_future_estimee_ha": surface_future,
        "evolution_surface_pct": evolution_surface_pct
    }


def build_harvest_department_ranking(
    df_harvest_score,
    top_n=10
):
    """
    Classe les départements selon leur rendement potentiel futur moyen,
    pondéré par la surface future estimée.
    """

    if df_harvest_score.empty:
        return pd.DataFrame()

    df = df_harvest_score.copy()

    df = df[
        df["department_code"].notna()
        & df["department_name"].notna()
    ].copy()

    if df.empty:
        return pd.DataFrame()

    resultats = []

    for (
        department_code,
        department_name
    ), groupe in df.groupby(
        ["department_code", "department_name"],
        dropna=True
    ):

        surface_future = float(
            groupe["surface_future_estimee_ha"].sum()
        )

        if surface_future <= 0:
            continue

        rendement_futur = moyenne_ponderee(
            groupe,
            "rendement_potentiel_q_ha",
            "surface_future_estimee_ha"
        )

        harvest_score = moyenne_ponderee(
            groupe,
            "harvest_score_futur",
            "surface_future_estimee_ha"
        )

        surface_reference = float(
            groupe["surface_reference_2024_ha"].sum()
        )

        rendement_reference = moyenne_ponderee(
            groupe.loc[groupe["historique_2024"]],
            "rendement_reference",
            "surface_reference_2024_ha"
        )

        if (
            pd.notna(rendement_futur)
            and pd.notna(rendement_reference)
            and rendement_reference > 0
        ):
            evolution_rendement_pct = (
                (
                    rendement_futur
                    - rendement_reference
                )
                / rendement_reference
                * 100
            )
        else:
            evolution_rendement_pct = np.nan

        resultats.append({
            "department_code": str(
                department_code
            ).strip(),
            "department_name": str(
                department_name
            ).strip(),
            "rendement_potentiel_q_ha": rendement_futur,
            "harvest_score": harvest_score,
            "surface_reference_2024_ha": surface_reference,
            "surface_future_estimee_ha": surface_future,
            "surface_maintenue_ha": float(
                groupe.loc[
                    groupe["statut"].eq("verte"),
                    "surface_future_estimee_ha"
                ].sum()
            ),
            "surface_perdue_ha": float(
                groupe.loc[
                    groupe["statut"].eq("rouge"),
                    "surface_reference_2024_ha"
                ].sum()
            ),
            "surface_nouvelle_estimee_ha": float(
                groupe.loc[
                    groupe["statut"].eq("bleue"),
                    "surface_future_estimee_ha"
                ].sum()
            ),
            "evolution_rendement_pct": (
                evolution_rendement_pct
            )
        })

    df_classement = pd.DataFrame(resultats)

    if df_classement.empty:
        return df_classement

    df_classement = (
        df_classement
        .sort_values(
            "rendement_potentiel_q_ha",
            ascending=False
        )
        .reset_index(drop=True)
    )

    df_classement["rang"] = (
        df_classement.index + 1
    )

    if top_n is not None:
        df_classement = df_classement.head(top_n)

    return df_classement

# ==========================================
# HEADER
# ==========================================

st.title("🌾 Harvest Games - Observatoire Agricole du futur")
st.image("assets/banner.png", use_container_width=True)
st.markdown(
    """
    <p style="font-size:22px;">
    Analysez l'impact du changement climatique sur la viabilité des cultures agricoles en France.
    Basé sur les données de l'INRAE et les projections CORDEX, Harvest Games permet d'envisager
    des scénarios au niveau local et départemental.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# 1. On définit le sélecteur ICI EN PREMIER (Global ou au début)
col1, col2, col3, col4 = st.columns([1, 2, 2, 2])

with col2:
    culture_sel_header = st.selectbox(
        "Sélectionnez la culture :",
        list(culture_info.keys()),
        key="culture_sel_h",
        format_func=lambda x: f"{culture_info[x]['picto']} {x}"
    )

current_culture_id = culture_info[culture_sel_header]["id"]
stats_nat = load_national_stats(current_culture_id)

stats_nat = load_national_stats(current_culture_id)
nb_parcelles = stats_nat['nb_parcelles'].values[0] if not stats_nat.empty else 0
total_ha_france = stats_nat['total_ha'].values[0] if not stats_nat.empty else 0

with col3:
    st.metric("Parcelles agricoles 2024", f"{nb_parcelles:,}".replace(",", " "))

with col4:
    st.metric("Surface totale en France en 2024", f"{total_ha_france:,.0f} ha".replace(",", " "))


ligne_contrainte_c1 = df_constraints[df_constraints['culture_name'] == culture_sel_header]
if not ligne_contrainte_c1.empty and pd.notna(ligne_contrainte_c1['temp_opt_max'].values[0]):
    seuil_alerte = float(ligne_contrainte_c1['temp_opt_max'].values[0])
else:
    seuil_alerte = 30.0

st.markdown("---")

# ==========================================
# CARTE 1 : SEUILS THERMIQUES
# ==========================================

col_cmd_1, col_carte_1 = st.columns([1, 3], gap="medium")

with col_cmd_1:
    for _ in range(7):
        st.write("")
        st.write("")

    st.markdown("### 🎛️ Paramètres")

    culture_sel_carte_1 = st.selectbox(
        "Sélectionnez la culture :",
        list(culture_info.keys()),
        key="culture_sel_c1",
        format_func=lambda x: f"{culture_info[x]['picto']} {x}"
    )

    mode_analyse = st.radio(
        "Mode d'analyse",
        [
            "Situation observée (2024)",
            "Projection CORDEX 2054",
            "Simulation de hausse"
        ],
        key="m1_carte1"
    )

    delta_temp = 0.0
    if mode_analyse == "Simulation de hausse":
        delta_temp = st.slider(
            "Hausse supplémentaire de température",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            key="delta_t1_carte1"
        )

# Le seuil dépend de la culture sélectionnée dans la Carte 1
ligne_contrainte_c1 = df_constraints[df_constraints["culture_name"] == culture_sel_carte_1]

if not ligne_contrainte_c1.empty and pd.notna(ligne_contrainte_c1["temp_opt_max"].iloc[0]):
    seuil_alerte = float(ligne_contrainte_c1["temp_opt_max"].iloc[0])
else:
    seuil_alerte = 30.0

culture_id_carte_1 = culture_info[
    culture_sel_carte_1
]["id"]

df_current = load_current_culture_data(
    culture_id=culture_id_carte_1,
    annee_cible=2054
)

with col_carte_1:
    if df_current.empty:
        st.warning("Aucune donnée trouvée pour cette culture dans la base Neon.")

    else:
        df_current = df_current.copy()
        MARGE_ORANGE = 2.0
        if mode_analyse == "Situation observée (2024)":
            df_current["temp_temperature"] = df_current["temp_2024"]

            titre_carte = (
                f"Situation observée en 2024 : "
                f"{culture_sel_carte_1} — seuil {seuil_alerte:.1f} °C"
            )

        elif mode_analyse == "Projection CORDEX 2054":
            df_current["temp_temperature"] = df_current["temp_2054"]

            titre_carte = (
                f"Projection CORDEX 2054 : "
                f"{culture_sel_carte_1} — seuil {seuil_alerte:.1f} °C"
            )

        else:
            df_current["temp_temperature"] = (
                df_current["temp_2054"] + delta_temp
            )

            titre_carte = (
                f"Projection CORDEX 2054 + {delta_temp:.1f} °C"
                f"{culture_sel_carte_1} — seuil {seuil_alerte:.1f} °C"
            )

        st.subheader("🌡️ Zones cultivées en 2024 menacées par la hausse des températures")

        st.caption(
            "ℹ️ CORDEX (Coordinated Regional Climate Downscaling Experiment) est un "
            "programme international de recherche produisant des projections "
            "climatiques régionalisées utilisées comme référence scientifique."
        )

        def definir_statut_thermique(temperature):
            if temperature <= seuil_alerte:
                return "Viable"
            elif temperature <= seuil_alerte + MARGE_ORANGE:
                return "Critique (Marge de 2°C)"
            else:
                return "Fatale"

        # Chaque mois reçoit un statut
        df_current["statut_mensuel"] = df_current["temp_temperature"].apply(definir_statut_thermique)

        ordre_statut = {
            "Viable": 1,
            "Critique (Marge de 2°C)": 2,
            "Fatale": 3
        }

        df_current["ordre"] = df_current["statut_mensuel"].map(ordre_statut)

        noms_mois = {
            4: "Avril",
            5: "Mai",
            6: "Juin",
            7: "Juillet",
            8: "Août"
        }

        df_current["nom_mois"] = df_current["month"].map(noms_mois)

        # Pour chaque zone, on conserve le mois le plus défavorable
        df_carte = (
            df_current
            .sort_values(
                by=["coordinates_id", "ordre", "temp_temperature"],
                ascending=[True, False, False]
            )
            .drop_duplicates(subset="coordinates_id", keep="first")
            .rename(columns={
                "statut_mensuel": "statut_parcelle",
                "nom_mois": "mois_defavorable"
            })
        )

        # Les surfaces exactes des parcelles sont comptées une seule fois par maille
        total_surface = df_carte["surface_ha_exacte"].sum()
        surf_opt = df_carte.loc[
            df_carte["statut_parcelle"] == "Viable",
            "surface_ha_exacte"
        ].sum()
        surf_tol = df_carte.loc[
            df_carte["statut_parcelle"] == "Critique (Marge de 2°C)",
            "surface_ha_exacte"
        ].sum()
        surf_crit = df_carte.loc[
            df_carte["statut_parcelle"] == "Fatale",
            "surface_ha_exacte"
        ].sum()

        pct_opt = surf_opt / total_surface * 100 if total_surface > 0 else 0
        pct_tol = surf_tol / total_surface * 100 if total_surface > 0 else 0
        pct_crit = surf_crit / total_surface * 100 if total_surface > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Surface analysée", f"{total_surface:,.0f} ha".replace(",", " "))
        m2.metric("Zones viables 🟢", f"{pct_opt:.2f} %")
        m3.metric("Zones critiques 🟠", f"{pct_tol:.2f} %")
        m4.metric("Zones fatales 🔴", f"{pct_crit:.2f} %")


        fig = px.scatter_mapbox(
            df_carte,
            lat="lat",
            lon="lon",
            color="statut_parcelle",
            size="surface_ha_exacte",
            size_max=12,
            color_discrete_map={
                "Viable": "#77dd77",
                "Critique (Marge de 2°C)": "#f39c12",
                "Fatale": "#e74c3c"
            },
            category_orders={
                "statut_parcelle": [
                    "Viable",
                    "Critique (Marge de 2°C)",
                    "Fatale"
                ]
            },
            mapbox_style="https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
            zoom=4.3,
            center={"lat": 46.2, "lon": 2.2},
            title=titre_carte,
            opacity=0.8,
            hover_data={
                "statut_parcelle": True,
                "surface_ha_exacte": ":,.2f",
                "nb_parcelles": True,
                "lat": False,
                "lon": False,
                "coordinates_id": False,
                "month": False,
                "ordre": False,
                "temp_2024": False,
                "temp_2054": False
            },
            labels={
                "statut_parcelle": "Statut",
                "surface_ha_exacte": "Surface cultivée exacte (ha)",
                "nb_parcelles": "Nombre de parcelles"
            }
        )

        fig.update_layout(
            height=600,
            margin={"r": 0, "t": 40, "l": 0, "b": 40},
            hoverlabel=dict(font_size=20),
            legend=dict(
                title=None,
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Méthode : les surfaces cultivées en 2024 proviennent des parcelles "
            "agricoles réelles de l'INRAE et du Ministère de l'Agriculture. "
            "Les projections futures CORDEX ont une résolution "
            "de 12,5 km : le statut affiché représente donc une tendance à l’échelle "
            "de la maille, et non une prévision individualisée pour chaque parcelle."
        )

        st.markdown(
            f"**Légende :** 🟢 jusqu'à {seuil_alerte:.1f} °C | "
            f"🟠 de {seuil_alerte:.1f} à {seuil_alerte + MARGE_ORANGE:.1f} °C | "
            f"🔴 au-delà de {seuil_alerte + MARGE_ORANGE:.1f} °C"
        )

        st.caption(
            "ℹ️ Les seuils sont évalués à partir des températures maximales observées sur plusieurs jours consécutifs de type canicule."
        )

st.markdown("---")

# ==========================================
# CARTE 2 - PROJECTIONS MULTICRITÈRES
# Saison culturale utilisée : avril à août inclus.
# ==========================================


# ==========================================
# 1. CHARGEMENT ET ÉVALUATION DES MAILLES
# ==========================================

@st.cache_data
def load_crop_projection(annee_sel, culture_id):
    """
    Évalue toutes les mailles utiles pour la culture et l'année choisies.

    Statuts retournés :
    - verte : culture présente en 2024 et tous les critères futurs sont respectés ;
    - rouge : culture présente en 2024 et au moins un critère futur échoue ;
    - bleue : culture absente en 2024 et tous les critères futurs sont respectés.

    Les valeurs de 2024 servent de référence informative dans les infobulles.
    Elles ne déterminent pas directement le statut futur.
    """

    query = """
        WITH contraintes AS (
            SELECT
                culture_id,
                temp_opt_min,
                temp_opt_max,
                precip_saison_min,
                precip_saison_max,
                rayonnement_saison_min_kwh_m2,
                rayonnement_saison_max_kwh_m2,
                preferred_soil_family_1,
                preferred_soil_family_2,
                preferred_soil_family_3
            FROM dim_constraints
            WHERE culture_id = :culture_id
        ),


        historique_2024 AS (
            SELECT
                coordinates_id,
                SUM(surf_ha) AS surface_ha_12_5km,
                COUNT(DISTINCT parcel_id) AS nb_parcelles_12_5km
            FROM src_cultures_plots_2024
            WHERE culture_id = :culture_id
            AND surf_ha > 0
            GROUP BY coordinates_id
        ),


        climat_2024_par_mois AS (
            SELECT
                coordinates_id,
                EXTRACT(MONTH FROM "date")::integer AS month,
                AVG(temp_mean_past) AS temp_mensuelle_2024,
                AVG(precip_past) AS precip_mensuelle_2024,
                AVG(solar_rad_past)
                    * 24
                    * EXTRACT(
                        DAY FROM (
                            MAKE_DATE(2024,
                                EXTRACT(MONTH FROM "date")::int,
                                1)
                            + INTERVAL '1 month'
                            - INTERVAL '1 day'
                        )
                    )
                    / 1000.0 AS solar_mensuel_2024_kwh_m2
            FROM src_climate_past
            WHERE "date" BETWEEN DATE '2024-04-01' AND DATE '2024-08-01'
            GROUP BY
                coordinates_id,
                EXTRACT(MONTH FROM "date")
        ),


        climat_2024_saison AS (
            SELECT
                coordinates_id,
                AVG(temp_mensuelle_2024) AS temp_moyenne_2024,
                MAX(temp_mensuelle_2024) AS temp_max_2024,
                SUM(precip_mensuelle_2024) AS precip_saison_2024,
                SUM(solar_mensuel_2024_kwh_m2) AS rayonnement_saison_2024,
                COUNT(*) AS nb_mois_2024
            FROM climat_2024_par_mois
            GROUP BY coordinates_id
        ),


        climat_futur_par_mois AS (
            SELECT
                coordinates_id,
                month,
                AVG(temp_mean_future) AS temp_mensuelle_future,
                AVG(precip_future) AS precip_mensuelle_future,
                AVG(solar_rad_future)
                    * 24
                    * EXTRACT(
                        DAY FROM (
                            MAKE_DATE(:annee_sel, month, 1)
                            + INTERVAL '1 month'
                            - INTERVAL '1 day'
                        )
                    )
                    / 1000.0 AS solar_mensuel_future_kwh_m2
            FROM src_climate_future
            WHERE year = :annee_sel
              AND month BETWEEN 4 AND 8
            GROUP BY
                coordinates_id,
                month
        ),


        climat_futur_saison AS (
            SELECT
                coordinates_id,
                AVG(temp_mensuelle_future) AS temp_moyenne_future,
                MAX(temp_mensuelle_future) AS temp_max_future,
                SUM(precip_mensuelle_future) AS precip_saison_future,
                SUM(solar_mensuel_future_kwh_m2) AS rayonnement_saison_future,
                COUNT(*) AS nb_mois_future
            FROM climat_futur_par_mois
            GROUP BY coordinates_id
        ),


        base_evaluation AS (
            SELECT
                coord.coordinates_id,
                coord.lat,
                coord.lon,

                sols.soil_family_type,
                sols.department_code,
                sols.department_name,

                COALESCE(hist.surface_ha_12_5km, 0) AS surface_ha_12_5km,
                COALESCE(hist.nb_parcelles_12_5km, 0) AS nb_parcelles_12_5km,
                (hist.coordinates_id IS NOT NULL) AS historique_2024,

                passe.temp_moyenne_2024,
                passe.temp_max_2024,
                passe.precip_saison_2024,
                passe.rayonnement_saison_2024,

                futur.temp_moyenne_future,
                futur.temp_max_future,
                futur.precip_saison_future,
                futur.rayonnement_saison_future,

                contraintes.temp_opt_min,
                contraintes.temp_opt_max,
                contraintes.precip_saison_min,
                contraintes.precip_saison_max,
                contraintes.rayonnement_saison_min_kwh_m2 / 3.6
                    AS rayonnement_saison_min_kwh_m2,
                contraintes.rayonnement_saison_max_kwh_m2 / 3.6
                    AS rayonnement_saison_max_kwh_m2,

                (
                    futur.nb_mois_future = 5
                    AND futur.temp_moyenne_future <= contraintes.temp_opt_max
                    AND futur.temp_max_future <= contraintes.temp_opt_max
                ) AS temp_ok,

                (
                    futur.nb_mois_future = 5
                    AND futur.precip_saison_future
                        BETWEEN contraintes.precip_saison_min
                        AND contraintes.precip_saison_max
                ) AS precip_ok,

                (
                    futur.nb_mois_future = 5
                    AND futur.rayonnement_saison_future
                        BETWEEN contraintes.rayonnement_saison_min_kwh_m2 / 3.6
                        AND contraintes.rayonnement_saison_max_kwh_m2 / 3.6
                ) AS solar_ok,

                (
                    sols.soil_family_type IN (
                        contraintes.preferred_soil_family_1,
                        contraintes.preferred_soil_family_2,
                        contraintes.preferred_soil_family_3
                    )
                ) AS soil_ok,

                (futur.nb_mois_future = 5) AS donnees_futures_completes

            FROM dim_coordinates coord

            JOIN dim_soils sols
                ON coord.coordinates_id = sols.coordinates_id

            JOIN climat_futur_saison futur
                ON coord.coordinates_id = futur.coordinates_id

            CROSS JOIN contraintes

            LEFT JOIN historique_2024 hist
                ON coord.coordinates_id = hist.coordinates_id

            LEFT JOIN climat_2024_saison passe
                ON coord.coordinates_id = passe.coordinates_id
        ),


        evaluation AS (
            SELECT
                *,

                (
                    COALESCE(temp_ok, FALSE)
                    AND COALESCE(precip_ok, FALSE)
                    AND COALESCE(solar_ok, FALSE)
                    AND COALESCE(soil_ok, FALSE)
                ) AS apte,

                CONCAT_WS(
                    ' + ',
                    CASE
                        WHEN NOT COALESCE(donnees_futures_completes, FALSE)
                        THEN 'données climatiques incomplètes'
                    END,
                    CASE
                        WHEN COALESCE(donnees_futures_completes, FALSE)
                         AND NOT COALESCE(temp_ok, FALSE)
                        THEN 'température'
                    END,
                    CASE
                        WHEN COALESCE(donnees_futures_completes, FALSE)
                         AND NOT COALESCE(precip_ok, FALSE)
                        THEN 'précipitations'
                    END,
                    CASE
                        WHEN COALESCE(donnees_futures_completes, FALSE)
                         AND NOT COALESCE(solar_ok, FALSE)
                        THEN 'rayonnement'
                    END,
                    CASE
                        WHEN NOT COALESCE(soil_ok, FALSE)
                        THEN 'sol'
                    END
                ) AS raison_echec

            FROM base_evaluation
        )


        SELECT
            coordinates_id,
            lat,
            lon,

            soil_family_type,
            department_code,
            department_name,

            surface_ha_12_5km,
            nb_parcelles_12_5km,
            historique_2024,

            temp_moyenne_2024,
            temp_max_2024,
            precip_saison_2024,
            rayonnement_saison_2024,

            temp_moyenne_future,
            temp_max_future,
            precip_saison_future,
            rayonnement_saison_future,

            temp_opt_min,
            temp_opt_max,
            precip_saison_min,
            precip_saison_max,
            rayonnement_saison_min_kwh_m2,
            rayonnement_saison_max_kwh_m2,

            temp_ok,
            precip_ok,
            solar_ok,
            soil_ok,
            apte,
            raison_echec,

            CASE
                WHEN historique_2024 AND apte
                    THEN 'verte'
                WHEN historique_2024 AND NOT apte
                    THEN 'rouge'
                WHEN NOT historique_2024 AND apte
                    THEN 'bleue'
                ELSE 'non_affichee'
            END AS statut

        FROM evaluation

        WHERE historique_2024 OR apte;
    """

    return conn.query(
        query,
        params={
            "annee_sel": int(annee_sel),
            "culture_id": int(culture_id)
        },
        ttl=0
    )

def build_department_vulnerability(
    df_projection,
    coefficient_nouvelles_zones=0.5
):
    """
    Agrège les résultats de la Carte 2 par département.

    Surface future estimée =
        surface historique maintenue
        + surface potentielle des nouvelles mailles aptes

    La surface potentielle d'une nouvelle maille est estimée à partir :
    - de la surface médiane historique par maille du département ;
    - ou, à défaut, de la médiane nationale ;
    - multipliée par un coefficient de prudence.
    """

    if df_projection.empty:
        return pd.DataFrame()

    colonnes_requises = {
        "coordinates_id",
        "department_code",
        "department_name",
        "surface_ha_12_5km",
        "historique_2024",
        "statut"
    }

    colonnes_manquantes = (
        colonnes_requises
        - set(df_projection.columns)
    )

    if colonnes_manquantes:
        raise ValueError(
            "Colonnes manquantes pour le calcul départemental : "
            f"{sorted(colonnes_manquantes)}"
        )

    df = df_projection.copy()

    # Nettoyage des codes et noms de départements
    df["department_code"] = (
        df["department_code"]
        .astype("string")
        .str.strip()
    )

    df["department_name"] = (
        df["department_name"]
        .astype("string")
        .str.strip()
    )

    # On exclut les mailles sans rattachement départemental.
    df = df[
        df["department_code"].notna()
        & df["department_name"].notna()
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # Sécurisation des surfaces
    df["surface_ha_12_5km"] = pd.to_numeric(
        df["surface_ha_12_5km"],
        errors="coerce"
    ).fillna(0)

    # Une ligne par maille seulement
    df = df.drop_duplicates(
        subset=["coordinates_id"],
        keep="first"
    )

    # --------------------------------------
    # 1. Surface médiane historique nationale
    # --------------------------------------

    surfaces_historiques = df.loc[
        df["historique_2024"],
        "surface_ha_12_5km"
    ]

    surface_mediane_nationale = (
        surfaces_historiques.median()
        if not surfaces_historiques.empty
        else 0
    )


    # --------------------------------------
    # 2. Médiane historique par département
    # --------------------------------------

    medianes_departements = (
        df.loc[
            df["historique_2024"]
            & (df["surface_ha_12_5km"] > 0)
        ]
        .groupby(
            ["department_code", "department_name"],
            as_index=False
        )
        .agg(
            surface_mediane_historique_maille=(
                "surface_ha_12_5km",
                "median"
            )
        )
    )

    # --------------------------------------
    # 3. Agrégation des statuts
    # --------------------------------------

    df_departements = (
        df.groupby(
            ["department_code", "department_name"],
            as_index=False
        )
        .agg(
            surface_2024_ha=(
                "surface_ha_12_5km",
                lambda serie: serie[
                    df.loc[serie.index, "historique_2024"]
                ].sum()
            ),

            surface_maintenue_ha=(
                "surface_ha_12_5km",
                lambda serie: serie[
                    df.loc[serie.index, "statut"] == "verte"
                ].sum()
            ),

            surface_perdue_ha=(
                "surface_ha_12_5km",
                lambda serie: serie[
                    df.loc[serie.index, "statut"] == "rouge"
                ].sum()
            ),

            nb_mailles_2024=(
                "historique_2024",
                "sum"
            ),

            nb_mailles_maintenues=(
                "statut",
                lambda serie: (serie == "verte").sum()
            ),

            nb_mailles_perdues=(
                "statut",
                lambda serie: (serie == "rouge").sum()
            ),

            nb_nouvelles_mailles_aptes=(
                "statut",
                lambda serie: (serie == "bleue").sum()
            )
        )
    )

    # --------------------------------------
    # 4. Ajout des médianes départementales
    # --------------------------------------

    df_departements = df_departements.merge(
        medianes_departements,
        on=["department_code", "department_name"],
        how="left"
    )

    # Si un département n'avait aucune maille historique,
    # on utilise la médiane nationale.
    df_departements[
        "surface_mediane_historique_maille"
    ] = (
        df_departements[
            "surface_mediane_historique_maille"
        ]
        .fillna(surface_mediane_nationale)
    )

    # --------------------------------------
    # 5. Estimation des nouvelles surfaces
    # --------------------------------------

    df_departements["surface_nouvelle_estimee_ha"] = (
        df_departements["nb_nouvelles_mailles_aptes"]
        * df_departements[
            "surface_mediane_historique_maille"
        ]
        * coefficient_nouvelles_zones
    )

    df_departements["surface_future_estimee_ha"] = (
        df_departements["surface_maintenue_ha"]
        + df_departements["surface_nouvelle_estimee_ha"]
    )

    # --------------------------------------
    # 6. Gain ou perte
    # --------------------------------------

    df_departements["evolution_surface_ha"] = (
        df_departements["surface_future_estimee_ha"]
        - df_departements["surface_2024_ha"]
    )

    df_departements["evolution_surface_pct"] = (
        df_departements["evolution_surface_ha"]
        .div(
            df_departements["surface_2024_ha"]
            .replace(0, pd.NA)
        )
        * 100
    )

    # --------------------------------------
    # 7. Construction de l'indice
    # --------------------------------------

    # Part de la surface historique qui reste adaptée.
    df_departements["taux_maintien"] = (
        df_departements["surface_maintenue_ha"]
        .div(
            df_departements["surface_2024_ha"]
            .replace(0, pd.NA)
        )
        .fillna(0)
    )

    # Part estimée des nouvelles surfaces par rapport à 2024.
    df_departements["bonus_nouvelles"] = (
        df_departements["surface_nouvelle_estimee_ha"]
        .div(
            df_departements["surface_2024_ha"]
            .replace(0, pd.NA)
        )
        .fillna(0)
    )

    # Le bonus est plafonné à 25 points.
    df_departements["bonus_nouvelles"] = (
        df_departements["bonus_nouvelles"]
        .clip(lower=0, upper=0.25)
    )

    # Indice compris entre -100 et +100.
    df_departements["indice"] = (
        df_departements["taux_maintien"] * 100
        + df_departements["bonus_nouvelles"] * 100
        - 50
    )

    df_departements["indice"] = (
        df_departements["indice"]
        .clip(lower=-100, upper=100)
        .round(0)
        .astype(int)
    )

    # Les départements sans surface historique n'ont pas
    # de référence 2024 suffisamment fiable.
    df_departements["a_reference_2024"] = (
        df_departements["surface_2024_ha"] > 0
    )


    # --------------------------------------
    # 8. Classement de l'indice
    # --------------------------------------

    def classer_departement(indice):

        if indice >= 80:
            return "Très favorable"

        elif indice >= 60:
            return "En progression"

        elif indice >= 40:
            return "Plutôt favorable"

        elif indice >= 20:
            return "Stable"

        elif indice >= 0:
            return "Sous surveillance"

        else:
            return "Fortement vulnérable"


    df_departements["statut_departement"] = (
        df_departements["indice"]
        .apply(classer_departement)
    )


    # --------------------------------------
    # 9. Conversion de l'indice en étoiles
    # --------------------------------------

    def indice_to_stars(indice):

        if indice >= 80:
            return "⭐⭐⭐⭐⭐"

        elif indice >= 60:
            return "⭐⭐⭐⭐☆"

        elif indice >= 40:
            return "⭐⭐⭐☆☆"

        elif indice >= 20:
            return "⭐⭐☆☆☆"

        elif indice >= 0:
            return "⭐☆☆☆☆"

        else:
            return "☆☆☆☆☆"


    df_departements["etoiles"] = (
        df_departements["indice"]
        .apply(indice_to_stars)
    )

    return (
        df_departements
        .sort_values(
            "evolution_surface_pct",
            ascending=False,
            na_position="last"
        )
        .reset_index(drop=True)
    )

@st.cache_data
def load_national_temperature_evolution():
    query = text("""
        WITH temperatures AS (

            -- Moyenne nationale observée en 2024
            SELECT
                2024 AS year,
                AVG(temp_mean_past) AS temperature
            FROM src_yields_cordex_2010_2025
            WHERE year = 2024
              AND temp_mean_past IS NOT NULL

            UNION ALL

            -- Moyennes nationales futures
            SELECT
                year,
                AVG(temp_mean_future) AS temperature
            FROM src_climate_future
            WHERE year IN (2054, 2084, 2100)
              AND temp_mean_future IS NOT NULL
            GROUP BY year
        )

        SELECT
            year,
            temperature
        FROM temperatures
        ORDER BY year;
    """)

    with engine.connect() as connection:
        return pd.read_sql(query, connection)

@st.cache_data
def load_map2_national_temperature_evolution():
    query = """
        WITH past_by_cell AS (
            SELECT
                coordinates_id,
                AVG(temp_mean_past) AS temperature
            FROM src_climate_past
            WHERE year = 2024
              AND month BETWEEN 4 AND 8
              AND temp_mean_past IS NOT NULL
            GROUP BY coordinates_id
        ),

        future_by_cell AS (
            SELECT
                year,
                coordinates_id,
                AVG(temp_mean_future) AS temperature
            FROM src_climate_future
            WHERE year IN (2054, 2084, 2100)
              AND month BETWEEN 4 AND 8
              AND temp_mean_future IS NOT NULL
            GROUP BY
                year,
                coordinates_id
        )

        SELECT
            2024 AS year,
            AVG(temperature) AS temperature
        FROM past_by_cell

        UNION ALL

        SELECT
            year,
            AVG(temperature) AS temperature
        FROM future_by_cell
        GROUP BY year

        ORDER BY year;
    """

    return conn.query(
        query,
        ttl=0
    )

@st.cache_data
def load_map2_local_temperature_evolution(coordinates_id):
    query = """
        SELECT
            year,
            temperature
        FROM (
            SELECT
                2024 AS year,
                AVG(temp_mean_past) AS temperature
            FROM src_climate_past
            WHERE year = 2024
              AND month BETWEEN 4 AND 8
              AND coordinates_id = :coordinates_id
              AND temp_mean_past IS NOT NULL

            UNION ALL

            SELECT
                year,
                AVG(temp_mean_future) AS temperature
            FROM src_climate_future
            WHERE year IN (2054, 2084, 2100)
              AND month BETWEEN 4 AND 8
              AND coordinates_id = :coordinates_id
              AND temp_mean_future IS NOT NULL
            GROUP BY year
        ) AS temperature_evolution

        ORDER BY year;
    """

    return conn.query(
        query,
        params={
            "coordinates_id": int(coordinates_id)
        },
        ttl=0
    )

# ==========================================
# 2. COLONNES DE LA CARTE
# ==========================================

col_param, col_carte = st.columns(
    [1, 3],
    gap="medium"
)

# ==========================================
# 3. PARAMÈTRES
# ==========================================

with col_param:

    for _ in range(8):
        st.write("")
        st.write("")
        st.write("")

    st.subheader("🎛️ Paramètres")

    culture_sel_carte_2 = st.selectbox(
        "Sélectionnez la culture :",
        list(culture_info.keys()),
        key="culture_sel_c2",
        format_func=lambda x: (
            f"{culture_info[x]['picto']} {x}"
        )
    )

    annee_sel = st.radio(
        "Scénario de projection",
        [2054, 2084, 2100],
        horizontal=True,
        key="annee_sel_carte_2"
    )


# ==========================================
# 4. IDENTIFIANT DE LA CULTURE
# ==========================================

current_culture_id_c2 = culture_info[
    culture_sel_carte_2
]["id"]


# ==========================================
# 5. CHARGEMENT ET CONTRÔLES
# ==========================================

with col_carte:

    st.subheader("🗺️ Cartographie des opportunités")

    st.caption(
        "Évaluation multicritère sur la saison culturale "
        "d’avril à août : température, précipitations, "
        "rayonnement et famille de sol."
    )

    try:
        df_projection = load_crop_projection(
            annee_sel=annee_sel,
            culture_id=current_culture_id_c2
        )

    except Exception as error:
        st.error(
            "La requête de la Carte 2 n’a pas pu être exécutée."
        )
        st.exception(error)
        st.stop()


    if df_projection.empty:
        st.warning(
            "Aucune maille historique ou future apte n’a été trouvée "
            "pour cette culture et cette échéance."
        )
        st.stop()

    df_departements = build_department_vulnerability(
    df_projection=df_projection,
    coefficient_nouvelles_zones=0.5
    )

    # Conversion défensive des booléens.
    for colonne_bool in [
        "historique_2024",
        "temp_ok",
        "precip_ok",
        "solar_ok",
        "soil_ok",
        "apte"
    ]:
        if colonne_bool in df_projection.columns:
            df_projection[colonne_bool] = (
                df_projection[colonne_bool]
                .fillna(False)
                .astype(bool)
            )


    df_green_plot = df_projection[
        df_projection["statut"] == "verte"
    ].copy()

    df_red_plot = df_projection[
        df_projection["statut"] == "rouge"
    ].copy()

    df_blue_plot = df_projection[
        df_projection["statut"] == "bleue"
    ].copy()


    # ======================================
    # 6. MÉTRIQUES
    # ======================================

    nb_zones_2024 = (
        df_projection.loc[
            df_projection["historique_2024"],
            "coordinates_id"
        ].nunique()
    )

    surf_totale_2024 = (
        df_projection.loc[
            df_projection["historique_2024"],
            "surface_ha_12_5km"
        ].sum()
    )

    zones_maintenues = (
        df_green_plot["coordinates_id"].nunique()
    )

    surf_maintenues = (
        df_green_plot["surface_ha_12_5km"].sum()
    )

    zones_perdues = (
        df_red_plot["coordinates_id"].nunique()
    )

    surf_perdues = (
        df_red_plot["surface_ha_12_5km"].sum()
    )

    zones_nouvelles = (
        df_blue_plot["coordinates_id"].nunique()
    )

    surfaces_historiques = df_projection.loc[
        df_projection["historique_2024"],
        "surface_ha_12_5km"
    ].dropna()

    surface_mediane_par_maille = (
        surfaces_historiques.median()
        if not surfaces_historiques.empty
        else 0
    )

    coefficient_ponderation = 0.5

    surface_estimee_nouvelles = (
        zones_nouvelles
        * surface_mediane_par_maille
        * coefficient_ponderation
    )


    # Une maille bleue représente une zone potentielle, pas une surface
    # déjà cultivée. On évite donc de transformer automatiquement son
    # nombre en hectares dans cette première version.
    m1, m2, m3 = st.columns(3)

    surface_future_estimee = (
        surf_maintenues
        + surface_estimee_nouvelles
    )

    # Évolution nette par rapport à la surface cultivée en 2024
    evolution_surface_ha = (
        surface_future_estimee
        - surf_totale_2024
    )

    evolution_surface_pct = (
        evolution_surface_ha
        / surf_totale_2024
        * 100
        if surf_totale_2024 > 0
        else 0
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Zones historiques maintenues",
        f"{surf_maintenues:,.0f} ha".replace(",", " "),
        help=(
            "Zones cultivées en 2024 qui respectent encore "
            f"tous les critères en {annee_sel}."
        )
    )

    m1.caption(
        f"{zones_maintenues:,} zones".replace(",", " ")
    )


    m2.metric(
        f"Zones historiques perdues en {annee_sel}",
        f"{surf_perdues:,.0f} ha".replace(",", " "),
        help=(
            "Zones cultivées en 2024 qui ne respectent plus "
            "au moins un des critères."
        )
    )

    m2.caption(
        f"{zones_perdues:,} zones".replace(",", " ")
    )


    m3.metric(
        f"Nouvelles zones aptes en {annee_sel}",
        f"≈ {surface_estimee_nouvelles:,.0f} ha".replace(",", " "),
        help=(
            "Surface indicative estimée à partir du nombre de nouvelles "
            "mailles aptes, de la surface médiane cultivée par maille en 2024 "
            "et d’un coefficient de pondération de 0,5."
        )
    )

    m3.caption(
        f"{zones_nouvelles:,} zones".replace(",", " ")
    )

    m4.metric(
        f"Évolution nette en {annee_sel}",
        f"{evolution_surface_pct:+.1f} %",
        delta=(
            f"{evolution_surface_ha:+,.0f} ha"
        ).replace(",", " "),
        help=(
            "Évolution calculée par rapport à la surface totale cultivée "
            "en 2024 : surfaces maintenues + nouvelles surfaces estimées."
        )
    )

    st.caption(
        (
            f"Référence 2024 : {nb_zones_2024:,} zones cultivées "
            f"— {surf_totale_2024:,.0f} ha."
        ).replace(",", " ")
    )


    # ======================================
    # 7. OUTILS D’INFOBULLE
    # ======================================

    def ajouter_colonnes_hover(df):
        """
        Prépare des textes lisibles sans modifier les calculs.
        """
        df = df.copy()

        df["raison_echec_affichee"] = (
            df["raison_echec"]
            .fillna("")
            .replace("", "aucune")
        )

        df["temp_ok_txt"] = df["temp_ok"].map(
            {True: "✅", False: "❌"}
        )
        df["precip_ok_txt"] = df["precip_ok"].map(
            {True: "✅", False: "❌"}
        )
        df["solar_ok_txt"] = df["solar_ok"].map(
            {True: "✅", False: "❌"}
        )
        df["soil_ok_txt"] = df["soil_ok"].map(
            {True: "✅", False: "❌"}
        )

        return df


    df_green_plot = ajouter_colonnes_hover(df_green_plot)
    df_red_plot = ajouter_colonnes_hover(df_red_plot)
    df_blue_plot = ajouter_colonnes_hover(df_blue_plot)


    # ======================================
    # 8. PRÉPARATION DES TAILLES DE POINTS
    # ======================================

    surface_max_historique = df_projection.loc[
        df_projection["historique_2024"],
        "surface_ha_12_5km"
    ].max()

    def taille_historique(df, taille_min=1.5, taille_max=10):

        if df.empty or surface_max_historique <= 0:
            return pd.Series(dtype=float)

        return (
            taille_min
            + (
                df["surface_ha_12_5km"]
                / surface_max_historique
            ) ** 0.5
            * (taille_max - taille_min)
        )

    df_green_plot["taille_point"] = taille_historique(df_green_plot)
    df_red_plot["taille_point"] = taille_historique(df_red_plot)


    # ======================================
    # 9. CARTE PLOTLY
    # ======================================

    fig = go.Figure()


    # --------------------------------------
    # Points bleus au fond
    # --------------------------------------

    if not df_blue_plot.empty:

        fig.add_trace(
            go.Scattermap(
                lat=df_blue_plot["lat"],
                lon=df_blue_plot["lon"],
                mode="markers",
                name=f"Nouvelle zone apte en {annee_sel}",

                marker=dict(
                    size=8,
                    color="#3498db",
                    opacity=1
                ),

                customdata=df_blue_plot[
                    [
                        "soil_family_type",
                        "temp_moyenne_future",
                        "temp_max_future",
                        "temp_opt_min",
                        "temp_opt_max",
                        "precip_saison_future",
                        "precip_saison_min",
                        "precip_saison_max",
                        "rayonnement_saison_future",
                        "rayonnement_saison_min_kwh_m2",
                        "rayonnement_saison_max_kwh_m2",
                        "temp_ok_txt",
                        "precip_ok_txt",
                        "solar_ok_txt",
                        "soil_ok_txt",
                        "coordinates_id"
                    ]
                ],

                hovertemplate=(
                    "<b>Nouvelle zone apte</b><br>"
                    "Sol : %{customdata[0]} %{customdata[14]}<br><br>"

                    "Température moyenne : %{customdata[1]:.1f} °C "
                    "[%{customdata[3]:.1f}–%{customdata[4]:.1f}] "
                    "%{customdata[11]}<br>"

                    "Mois le plus chaud : %{customdata[2]:.1f} °C "
                    "(maximum autorisé : %{customdata[4]:.1f})<br>"

                    "Précipitations : %{customdata[5]:.0f} mm "
                    "[%{customdata[6]:.0f}–%{customdata[7]:.0f}] "
                    "%{customdata[12]}<br>"

                    "Rayonnement : %{customdata[8]:.0f} kWh/m² "
                    "[%{customdata[9]:.0f}–%{customdata[10]:.0f}] "
                    "%{customdata[13]}"
                    "<extra></extra>"
                )
            )
        )


    # --------------------------------------
    # Points verts
    # --------------------------------------

    if not df_green_plot.empty:

        fig.add_trace(
            go.Scattermap(
                lat=df_green_plot["lat"],
                lon=df_green_plot["lon"],
                mode="markers",
                name="Zone historique maintenue",

                marker=dict(
                    size=df_green_plot["taille_point"],
                    color="#2ecc71",
                    opacity=0.85
                ),

                customdata=df_green_plot[
                    [
                        "surface_ha_12_5km",
                        "soil_family_type",
                        "temp_moyenne_2024",
                        "temp_moyenne_future",
                        "temp_max_future",
                        "temp_opt_min",
                        "temp_opt_max",
                        "precip_saison_2024",
                        "precip_saison_future",
                        "precip_saison_min",
                        "precip_saison_max",
                        "rayonnement_saison_2024",
                        "rayonnement_saison_future",
                        "rayonnement_saison_min_kwh_m2",
                        "rayonnement_saison_max_kwh_m2",
                        "temp_ok_txt",
                        "precip_ok_txt",
                        "solar_ok_txt",
                        "soil_ok_txt",
                        "coordinates_id"
                    ]
                ],

                hovertemplate=(
                    "<b>Zone historique maintenue</b><br>"
                    "Surface cultivée en 2024 : "
                    "%{customdata[0]:,.0f} ha<br>"
                    "Sol : %{customdata[1]} %{customdata[18]}<br><br>"

                    "Température moyenne 2024 : "
                    "%{customdata[2]:.1f} °C<br>"
                    f"Température moyenne {annee_sel} : "
                    "%{customdata[3]:.1f} °C "
                    "[%{customdata[5]:.1f}–%{customdata[6]:.1f}] "
                    "%{customdata[15]}<br>"
                    f"Mois le plus chaud en {annee_sel} : "
                    "%{customdata[4]:.1f} °C<br><br>"

                    "Précipitations 2024 : "
                    "%{customdata[7]:.0f} mm<br>"
                    f"Précipitations {annee_sel} : "
                    "%{customdata[8]:.0f} mm "
                    "[%{customdata[9]:.0f}–%{customdata[10]:.0f}] "
                    "%{customdata[16]}<br><br>"

                    "Rayonnement 2024 : "
                    "%{customdata[11]:.0f} kWh/m²<br>"
                    f"Rayonnement {annee_sel} : "
                    "%{customdata[12]:.0f} kWh/m² "
                    "[%{customdata[13]:.0f}–%{customdata[14]:.0f}] "
                    "%{customdata[17]}"
                    "<extra></extra>"
                )
            )
        )


    # --------------------------------------
    # Points rouges au-dessus
    # --------------------------------------

    if not df_red_plot.empty:

        fig.add_trace(
            go.Scattermap(
                lat=df_red_plot["lat"],
                lon=df_red_plot["lon"],
                mode="markers",
                name=f"Zone historique perdue en {annee_sel}",

                marker=dict(
                    size=df_red_plot["taille_point"],
                    color="#e74c3c",
                    opacity=0.35
                ),

                customdata=df_red_plot[
                    [
                        "surface_ha_12_5km",
                        "soil_family_type",
                        "temp_moyenne_future",
                        "temp_max_future",
                        "temp_opt_min",
                        "temp_opt_max",
                        "precip_saison_future",
                        "precip_saison_min",
                        "precip_saison_max",
                        "rayonnement_saison_future",
                        "rayonnement_saison_min_kwh_m2",
                        "rayonnement_saison_max_kwh_m2",
                        "temp_ok_txt",
                        "precip_ok_txt",
                        "solar_ok_txt",
                        "soil_ok_txt",
                        "raison_echec_affichee",
                        "coordinates_id"
                    ]
                ],

                hovertemplate=(
                    "<b>Zone historique perdue</b><br>"
                    "Surface cultivée en 2024 : "
                    "%{customdata[0]:,.0f} ha<br>"
                    "<b>Critères en échec : %{customdata[16]}</b><br><br>"

                    "Sol : %{customdata[1]} %{customdata[15]}<br>"

                    "Température moyenne : %{customdata[2]:.1f} °C "
                    "[%{customdata[4]:.1f}–%{customdata[5]:.1f}] "
                    "%{customdata[12]}<br>"

                    "Mois le plus chaud : %{customdata[3]:.1f} °C "
                    "(maximum autorisé : %{customdata[5]:.1f})<br>"

                    "Précipitations : %{customdata[6]:.0f} mm "
                    "[%{customdata[7]:.0f}–%{customdata[8]:.0f}] "
                    "%{customdata[13]}<br>"

                    "Rayonnement : %{customdata[9]:.0f} kWh/m² "
                    "[%{customdata[10]:.0f}–%{customdata[11]:.0f}] "
                    "%{customdata[14]}"
                    "<extra></extra>"
                )
            )
        )


    # --------------------------------------
    # Mise en forme
    # --------------------------------------

    fig.update_layout(
        clickmode="event+select",
        hoverlabel=dict(font_size=20),
        map=dict(
            style=(
                "https://basemaps.cartocdn.com/gl/"
                "positron-nolabels-gl-style/style.json"
            ),
            zoom=4.3,
            center={
                "lat": 46.2,
                "lon": 2.2
            }
        ),

        title=(
            f"Projection {annee_sel} : opportunités pour "
            f"{culture_sel_carte_2}"
        ),

        height=650,

        margin={
            "r": 0,
            "t": 45,
            "l": 0,
            "b": 90
        },

        legend={
            "title": {
                "text": "Statut de projection"
            },
            "orientation": "h",
            "yanchor": "top",
            "y": -0.10,
            "xanchor": "center",
            "x": 0.5
        }
    )

    map_event = st.plotly_chart(
    fig,
    width="stretch",
    key="map2_projection",
    on_select="rerun",
    selection_mode="points",
    config={
        "scrollZoom": False
    }
    )

    selected_coordinates_id = None

    if map_event.selection.points:
        selected_point = map_event.selection.points[0]
        selected_customdata = selected_point.get("customdata")

        if selected_customdata:
            selected_coordinates_id = selected_customdata[-1]

    st.write("Maille sélectionnée :", selected_coordinates_id)

    if selected_coordinates_id is not None:

        selected_coordinates_id = int(selected_coordinates_id)

        df_temp_nationale = (
            load_map2_national_temperature_evolution()
            .copy()
        )

        df_temp_locale = (
            load_map2_local_temperature_evolution(
                selected_coordinates_id
            )
            .copy()
        )

        if df_temp_locale.empty:
            st.warning(
                "Aucune donnée de température disponible "
                "pour cette maille."
            )

        else:
            # Renommage des colonnes pour distinguer les deux courbes
            df_temp_nationale = df_temp_nationale.rename(
                columns={
                    "temperature": "Température moyenne France"
                }
            )

            df_temp_locale = df_temp_locale.rename(
                columns={
                    "temperature": "Température de la maille"
                }
            )

            # Fusion sur les années
            df_temperature_evolution = pd.merge(
                df_temp_nationale,
                df_temp_locale,
                on="year",
                how="outer"
            ).sort_values("year")

            # Conversion défensive des types
            df_temperature_evolution["year"] = (
                pd.to_numeric(
                    df_temperature_evolution["year"],
                    errors="coerce"
                )
            )

            df_temperature_evolution[
                "Température moyenne France"
            ] = pd.to_numeric(
                df_temperature_evolution[
                    "Température moyenne France"
                ],
                errors="coerce"
            )

            df_temperature_evolution[
                "Température de la maille"
            ] = pd.to_numeric(
                df_temperature_evolution[
                    "Température de la maille"
                ],
                errors="coerce"
            )

            # Création du graphique
            fig_temperature = go.Figure()

            fig_temperature.add_trace(
                go.Scatter(
                    x=df_temperature_evolution["year"],
                    y=df_temperature_evolution[
                        "Température moyenne France"
                    ],
                    mode="lines+markers",
                    name="Moyenne nationale",
                    line=dict(
                    color="#0057D9",
                    width=4,
                    dash="dashdot"
                ),
                marker=dict(
                    size=8,
                    color="#0057D9"
                )
            )
            )

            fig_temperature.add_trace(
                go.Scatter(
                    x=df_temperature_evolution["year"],
                    y=df_temperature_evolution[
                        "Température de la maille"
                    ],
                    mode="lines+markers",
                    name=f"Maille {selected_coordinates_id}",
                    line=dict(
                    color="#E53935",
                    width=3
                    ),
                    marker=dict(
                    size=8,
                    color="#E53935"
                )
            )
            )

            fig_temperature.update_layout(
                title=(
                    "Évolution de la température moyenne "
                    "d’avril à août"
                ),
                xaxis_title="Année",
                yaxis_title="Température moyenne (°C)",
                hovermode="x unified",
                height=420,
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 60,
                    "b": 20
                },
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5
                }
            )

            fig_temperature.update_xaxes(
                tickmode="array",
                tickvals=[2024, 2054, 2084, 2100]
            )

            st.plotly_chart(
                fig_temperature,
                width="stretch",
                key=(
                    f"temperature_evolution_"
                    f"{selected_coordinates_id}"
                )
            )

# ======================================
# MODULE 4 - DEPARTEMENTAL
# ======================================

st.markdown("---")

col_info_dep, col_carte_dep = st.columns(
    [1, 3],
    gap="medium"
)

# ==========================
# Colonne de gauche
# ==========================

with col_info_dep:

    for _ in range(4):
        st.write("")

    st.markdown("### 🎛️ Paramètres")

    culture_sel_dep = st.selectbox(
        "Sélectionnez la culture :",
        list(culture_info.keys()),
        key="culture_sel_departements",
        format_func=lambda x: (
            f"{culture_info[x]['picto']} {x}"
        )
    )

    annee_sel_dep = st.radio(
        "Sélection du scénario",
        [2054, 2084, 2100],
        horizontal=True,
        key="annee_sel_departements"
    )

    st.markdown("### 📊 Lecture de l'indice")

    st.markdown(
        """
        **Indice positif**  
        Le département conserve une part de ses surfaces cultivées
        et bénéficie de nouvelles zones.

        **Indice proche de zéro**  
        Les gains compensent partiellement les pertes.

        **Indice négatif**  
        Le département est exposé à une perte conséquente de parcelles.
        """
    )


# ==========================
# Données départementales
# ==========================

culture_id_dep = culture_info[
    culture_sel_dep
]["id"]

df_projection_dep = load_crop_projection(
    annee_sel=annee_sel_dep,
    culture_id=culture_id_dep
)

df_departements_dep = build_department_vulnerability(
    df_projection=df_projection_dep,
    coefficient_nouvelles_zones=0.5
)

geojson_departements = load_departments_geojson()

df_departements_dep["department_code"] = (
    df_departements_dep["department_code"]
    .astype(str)
    .str.strip()
    .str.zfill(2)
)

# ==========================
# Création de la carte
# ==========================

fig_departements = px.choropleth(
    data_frame=df_departements_dep,
    geojson=geojson_departements,
    locations="department_code",
    featureidkey="properties.code",
    color="indice",
    hover_name="department_name",
    hover_data={
        "department_code": False,
        "indice": ":.0f",
        "etoiles": True,
        "surface_2024_ha": ":,.0f",
        "surface_maintenue_ha": ":,.0f",
        "surface_nouvelle_estimee_ha": ":,.0f",
        "surface_future_estimee_ha": ":,.0f",
        "evolution_surface_ha": ":,.0f",
        "evolution_surface_pct": ":.1f",
        "statut_departement": True
    },
    labels={
        "indice": "Indice",
        "etoiles": "Évaluation",
        "surface_2024_ha": "Surface cultivée en 2024 (ha)",
        "surface_maintenue_ha": (
            f"Surface maintenue en {annee_sel_dep} (ha)"
        ),
        "surface_nouvelle_estimee_ha": (
            "Nouvelle surface estimée (ha)"
        ),
        "surface_future_estimee_ha": (
            f"Surface future estimée en {annee_sel_dep} (ha)"
        ),
        "evolution_surface_ha": "Évolution nette (ha)",
        "evolution_surface_pct": "Évolution nette (%)",
        "statut_departement": "Statut"
    },
    color_continuous_scale="RdYlGn",
    range_color=(-100, 100)
)

fig_departements.update_traces(
    hoverlabel=dict(font_size=20)
)

fig_departements.update_geos(
    fitbounds="locations",
    visible=False
)

fig_departements.update_layout(
    title=None,
    height=520,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),
    coloraxis_colorbar=dict(
        title="Indice",
        len=0.72,
        thickness=18,
        y=0.5,
        yanchor="middle"
    )
)

# ==========================
# Colonne de droite
# ==========================

with col_carte_dep:

    st.subheader(
        f"🏛️ Indice départemental de vulnérabilité — "
        f"{culture_sel_dep}"
    )

    st.caption(
        "L'indice départemental est construit à partir des résultats "
        f"des projections multicritères pour {annee_sel_dep}. "
        "Il combine le maintien des surfaces cultivées en 2024 et "
        "le potentiel de nouvelles zones adaptées afin de comparer "
        "la résilience future des départements. "
        "Les surfaces affichées concernent uniquement la culture sélectionnée."
    )

    st.plotly_chart(
    fig_departements,
    use_container_width=True,
    config={
        "scrollZoom": False,
        "displayModeBar": False
    }
    )


# ======================================
# MODULE 5 - HARVEST SCORE
# Synthèse nationale et Top 10 départemental
# ======================================

st.markdown("---")

col_param_score, col_resultat_score = st.columns(
    [1, 3],
    gap="medium"
)

with col_param_score:

    st.markdown("### 🎛️ Paramètres")

    culture_sel_score = st.selectbox(
        "Sélectionnez la culture :",
        list(culture_info.keys()),
        key="culture_sel_harvest_score",
        format_func=lambda x: (
            f"{culture_info[x]['picto']} {x}"
        )
    )

    annee_sel_score = st.radio(
        "Sélection du scénario",
        [2054, 2084, 2100],
        horizontal=True,
        key="annee_sel_harvest_score"
    )

    st.markdown("### 🧩 Composition du score")

    df_poids_score = pd.DataFrame({
        "Paramètre": [
            "Rendement potentiel",
            "Température",
            "Précipitations",
            "Rayonnement",
            "Sol"
        ],
        "Poids": [40, 25, 15, 10, 10]
    })

    fig_poids_score = px.pie(
        df_poids_score,
        names="Paramètre",
        values="Poids",
        hole=0.45
    )

    fig_poids_score.update_traces(
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Poids dans le score : %{value} %"
            "<extra></extra>"
        )
    )

    fig_poids_score.update_layout(
        height=360,
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=20
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(
        fig_poids_score,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


culture_id_score = culture_info[
    culture_sel_score
]["id"]

with col_resultat_score:

    st.subheader(
        f"🌾 Harvest Score — {culture_sel_score}"
    )

    st.caption(
        "Le Harvest Score national combine le potentiel agronomique "
        "des mailles et l'évolution des surfaces. Les zones historiques "
        "maintenues sont conservées, les nouvelles zones aptes sont "
        "estimées avec un coefficient prudent de 0,5 et les zones "
        "perdues ont une contribution future nulle."
    )

    calculer_harvest_score = st.button(
        "🌾 Calculer le Harvest Score",
        type="primary",
        key="btn_calcul_harvest_score"
    )

    if calculer_harvest_score:

        try:
            with st.spinner(
                "Calcul des rendements potentiels, des surfaces "
                "et du Harvest Score..."
            ):
                df_harvest_score, metriques_modele = (
                    build_harvest_score_data(
                        culture_id=culture_id_score,
                        annee_sel=annee_sel_score,
                        df_constraints=df_constraints,
                        coefficient_nouvelles_zones=0.5
                    )
                )

                indicateurs_nationaux = (
                    calculer_indicateurs_nationaux_harvest(
                        df_harvest_score
                    )
                )

                df_top_departements = (
                    build_harvest_department_ranking(
                        df_harvest_score,
                        top_n=None
                    )
                )

        except Exception as error:
            st.error(
                "Le Harvest Score n'a pas pu être calculé."
            )
            st.exception(error)

        else:
            st.success("Calcul terminé.")

            if (
                df_harvest_score.empty
                or not indicateurs_nationaux
            ):
                st.warning(
                    "Aucune donnée future complète n'est disponible "
                    "pour cette culture et ce scénario."
                )

            else:
                score_national = indicateurs_nationaux[
                    "harvest_score_national"
                ]
                rendement_national = indicateurs_nationaux[
                    "rendement_national_futur"
                ]
                production_future = indicateurs_nationaux[
                    "production_future_estimee_q"
                ]
                evolution_production = indicateurs_nationaux[
                    "evolution_production_pct"
                ]
                surface_future = indicateurs_nationaux[
                    "surface_future_estimee_ha"
                ]
                evolution_surface = indicateurs_nationaux[
                    "evolution_surface_pct"
                ]

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    f"Harvest Score national {annee_sel_score}",
                    (
                        f"{score_national:.1f} / 100"
                        if pd.notna(score_national)
                        else "Non disponible"
                    ),
                    help=(
                        "Les zones rouges restent dans la surface "
                        "de référence mais contribuent avec zéro point. "
                        "Les zones bleues sont intégrées avec une surface "
                        "estimée et un coefficient de prudence de 0,5."
                    )
                )

                m2.metric(
                    f"Production nationale potentielle en {annee_sel_score}",
                    (
                        f"{production_future:,.0f} q".replace(",", " ")
                        if pd.notna(production_future)
                        else "Non disponible"
                    ),
                    delta=(
                        f"{evolution_production:+.1f} % vs 2024"
                        if pd.notna(evolution_production)
                        else None
                    ),
                    help=(
                        "Estimation obtenue en multipliant le rendement "
                        "potentiel national pondéré par la surface future "
                        "estimée. Elle est comparée à la production "
                        "potentielle de référence en 2024."
                    )
                )

                m3.metric(
                    f"Surface future estimée en {annee_sel_score}",
                    (
                        f"{surface_future:,.0f} ha"
                        .replace(",", " ")
                    ),
                    delta=(
                        f"{evolution_surface:+.1f} % vs 2024"
                        if pd.notna(evolution_surface)
                        else None
                    ),
                    help=(
                        "Surface historique maintenue + nouvelles "
                        "surfaces estimées. Les zones rouges sont exclues "
                        "de la surface productive future."
                    )
                )

                st.caption(
                    "Les indicateurs sont comparés à la situation agricole de référence en 2024."
                )

                if metriques_modele["mae"] is not None:
                    st.caption(
                        f"Modèle entraîné sur "
                        f"{metriques_modele['nb_lignes']:,} observations."
                        .replace(",", " ")
                    )

                st.info(
                    "Le classement ci-dessous présente l’ensemble des départements "
                    f"disposant d’une surface future estimée positive en {annee_sel_score}. "
                    "Le tableau peut être trié en cliquant sur l’en-tête de chaque colonne."
                )

                st.subheader(
                    "🏆 Classement des départements par rendement potentiel"
                )

                if df_top_departements.empty:
                    st.warning(
                        "Aucun département ne dispose d'une surface "
                        "future estimée positive."
                    )

                else:
                    df_affichage_top = (
                        df_top_departements[
                            [
                                "rang",
                                "department_name",
                                "rendement_potentiel_q_ha",
                                "harvest_score",
                                "surface_future_estimee_ha",
                                "evolution_rendement_pct"
                            ]
                        ]
                        .rename(
                            columns={
                                "rang": "Rang",
                                "department_name": "Département",
                                "rendement_potentiel_q_ha": (
                                    "Rendement potentiel (q/ha)"
                                ),
                                "harvest_score": (
                                    "Harvest Score (/100)"
                                ),
                                "surface_future_estimee_ha": (
                                    "Surface future estimée (ha)"
                                ),
                                "evolution_rendement_pct": (
                                    "Évolution du rendement (%)"
                                )
                            }
                        )
                    )

                    st.dataframe(
                        df_affichage_top,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rang": st.column_config.NumberColumn(
                                format="%d"
                            ),
                            "Rendement potentiel (q/ha)": (
                                st.column_config.NumberColumn(
                                    format="%.1f"
                                )
                            ),
                            "Harvest Score (/100)": (
                                st.column_config.NumberColumn(
                                    format="%.1f"
                                )
                            ),
                            "Surface future estimée (ha)": (
                                st.column_config.NumberColumn(
                                    format="%.0f"
                                )
                            ),
                            "Évolution du rendement (%)": (
                                st.column_config.NumberColumn(
                                    format="%+.1f %%"
                                )
                            )
                        }
                    )

                with st.expander(
                    "🔎 Contrôler les calculs par statut"
                ):
                    resume_statuts = (
                        df_harvest_score
                        .groupby(
                            "statut",
                            as_index=False
                        )
                        .agg(
                            nb_mailles=(
                                "coordinates_id",
                                "nunique"
                            ),
                            surface_reference_2024_ha=(
                                "surface_reference_2024_ha",
                                "sum"
                            ),
                            surface_future_estimee_ha=(
                                "surface_future_estimee_ha",
                                "sum"
                            ),
                            rendement_potentiel_moyen_q_ha=(
                                "rendement_potentiel_q_ha",
                                "mean"
                            ),
                            harvest_score_moyen=(
                                "harvest_score_futur",
                                "mean"
                            )
                        )
                    )

