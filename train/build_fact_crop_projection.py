"""Construction de la table de faits dans Neon.

1. charge les données historiques et futures depuis Neon ;
2. entraîne un RandomForestRegressor une fois par culture ;
3. calcule les statuts agronomiques et le Harvest Score par maille ;
4. remplace atomiquement chaque partition culture × année dans la table de faits.

Variables d'environnement obligatoires :
    NEON_DATABASE_URL

Variable facultative :
    RAYONNEMENT_CONSTRAINT_DIVISOR (3.6 par défaut, comme l'application actuelle)
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import Engine, create_engine, text


CULTURE_IDS = (1, 2, 3)
PROJECTION_YEARS = (2054, 2084, 2100)
COEFFICIENT_NOUVELLES_ZONES = 0.5
MODEL_VERSION = "random_forest_v1"

FEATURES_RENDEMENT_NUMERIQUES = [
    "temp_saison",
    "temp_max_saison",
    "precip_cumul",
    "solar_moy",
]
FEATURES_RENDEMENT_CATEGORIELLES = ["soil_family_type"]

POIDS_HARVEST_SCORE = {
    "rendement": 0.15,
    "temperature": 0.35,
    "precipitations": 0.20,
    "rayonnement": 0.15,
    "sol": 0.15,
}

FACT_COLUMNS = [
    "coordinates_id",
    "culture_id",
    "projection_year",
    "department_code",
    "department_name",
    "region_code",
    "region_name",
    "lat",
    "lon",
    "soil_family_type",
    "soil_type",
    "historique_2024",
    "nb_parcelles_2024",
    "surface_reference_2024_ha",
    "temp_moyenne_2024",
    "temp_max_2024",
    "precip_saison_2024",
    "rayonnement_saison_2024_kwh_m2",
    "temp_moyenne_future",
    "temp_max_future",
    "precip_saison_future",
    "rayonnement_saison_future_kwh_m2",
    "temp_ok",
    "precip_ok",
    "solar_ok",
    "soil_ok",
    "donnees_futures_completes",
    "apte",
    "statut",
    "raison_echec",
    "rendement_reference_q_ha",
    "rendement_potentiel_q_ha",
    "score_rendement_reference",
    "score_rendement_futur",
    "score_temperature_reference",
    "score_temperature_futur",
    "score_precip_reference",
    "score_precip_futur",
    "score_rayonnement_reference",
    "score_rayonnement_futur",
    "score_sol",
    "harvest_score_reference",
    "harvest_score_futur",
    "evolution_score_points",
    "evolution_rendement_pct",
    "surface_mediane_historique_maille_ha",
    "surface_future_estimee_ha",
    "surface_nouvelle_estimee_ha",
    "model_version",
    "coefficient_nouvelles_zones",
]


@dataclass
class ModelBundle:
    model: Pipeline
    rendement_reference_haut: float
    reference_par_maille: pd.DataFrame
    valeurs_reference_culture: dict[str, float]
    metrics: dict[str, float | int | None]


def get_database_url() -> str:
    database_url = os.getenv("NEON_DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "NEON_DATABASE_URL est absente. Ajoute-la dans l'environnement "
            "ou dans le service Docker Compose."
        )
    return database_url


def get_rayonnement_divisor() -> float:
    raw_value = os.getenv("RAYONNEMENT_CONSTRAINT_DIVISOR", "3.6")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "RAYONNEMENT_CONSTRAINT_DIVISOR doit être un nombre."
        ) from exc
    if value <= 0:
        raise RuntimeError(
            "RAYONNEMENT_CONSTRAINT_DIVISOR doit être strictement positif."
        )
    return value


def make_engine(database_url: str) -> Engine:
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "sslmode": "require",
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )


def read_dataframe(
    engine: Engine,
    query: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection, params=params or {})


def load_constraints(engine: Engine) -> pd.DataFrame:
    return read_dataframe(
        engine,
        "SELECT * FROM public.dim_constraints ORDER BY culture_id;",
    )


def load_yield_training_data(engine: Engine, culture_id: int) -> pd.DataFrame:
    query = """
        WITH climat_historique_mensuel AS (
            SELECT
                coordinates_id,
                year,
                month,
                AVG(temp_mean_past) AS temp_mois,
                AVG(precip_past) AS precip_mois,
                AVG(solar_rad_past) AS solar_mois
            FROM public.src_climate_past
            WHERE year BETWEEN 2010 AND 2025
              AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, year, month
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
            GROUP BY coordinates_id, year
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
            COALESCE(sols.soil_family_type, 'inconnu') AS soil_family_type
        FROM public.src_yields_cordex_2010_2025 AS rendements
        JOIN climat_historique_saison AS climat
          ON rendements.coordinates_id = climat.coordinates_id
         AND rendements.year = climat.year
        LEFT JOIN public.dim_soils AS sols
          ON rendements.coordinates_id = sols.coordinates_id
        WHERE rendements.culture_id = :culture_id
          AND rendements.rendement_q_ha IS NOT NULL
          AND rendements.rendement_q_ha > 0
          AND COALESCE(rendements.surface_ha_12_5km, 0) > 0
          AND climat.nb_mois = 5;
    """
    return read_dataframe(engine, query, {"culture_id": int(culture_id)})


def load_future_yield_features(engine: Engine, projection_year: int) -> pd.DataFrame:
    query = """
        WITH climat_futur_mensuel AS (
            SELECT
                coordinates_id,
                year,
                month,
                AVG(temp_mean_future) AS temp_mois,
                AVG(precip_future) AS precip_mois,
                AVG(solar_rad_future) AS solar_mois
            FROM public.src_climate_future
            WHERE year = :projection_year
              AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, year, month
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
            GROUP BY coordinates_id, year
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
            COALESCE(sols.soil_family_type, 'inconnu') AS soil_family_type
        FROM climat_futur_saison AS climat
        JOIN public.dim_coordinates AS coord
          ON climat.coordinates_id = coord.coordinates_id
        LEFT JOIN public.dim_soils AS sols
          ON climat.coordinates_id = sols.coordinates_id
        WHERE climat.nb_mois = 5;
    """
    return read_dataframe(
        engine,
        query,
        {"projection_year": int(projection_year)},
    )


def load_crop_projection(
    engine: Engine,
    projection_year: int,
    culture_id: int,
    rayonnement_divisor: float,
) -> pd.DataFrame:
    """Évalue les mailles historiques ou futures aptes pour une culture."""

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
            FROM public.dim_constraints
            WHERE culture_id = :culture_id
        ),
        historique_2024 AS (
            SELECT
                coordinates_id,
                SUM(surf_ha) AS surface_ha_12_5km,
                COUNT(DISTINCT parcel_id) AS nb_parcelles_12_5km
            FROM public.vw_cultures_plots_2024_clean
            WHERE culture_id = :culture_id
              AND surf_ha > 0
            GROUP BY coordinates_id
        ),
        climat_2024_par_mois AS (
            SELECT
                coordinates_id,
                month,
                AVG(temp_mean_past) AS temp_mensuelle_2024,
                AVG(precip_past) AS precip_mensuelle_2024,
                AVG(solar_rad_past)
                    * 24
                    * EXTRACT(
                        DAY FROM (
                            MAKE_DATE(2024, month, 1)
                            + INTERVAL '1 month'
                            - INTERVAL '1 day'
                        )
                    )
                    / 1000.0 AS solar_mensuel_2024_kwh_m2
            FROM public.src_climate_past
            WHERE year = 2024
              AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, month
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
                            MAKE_DATE(:projection_year, month, 1)
                            + INTERVAL '1 month'
                            - INTERVAL '1 day'
                        )
                    )
                    / 1000.0 AS solar_mensuel_future_kwh_m2
            FROM public.src_climate_future
            WHERE year = :projection_year
              AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, month
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
                sols.soil_type,
                sols.department_code,
                sols.department_name,
                sols.region_code,
                sols.region_name,
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
                (
                    futur.nb_mois_future = 5
                    AND futur.temp_moyenne_future
                        BETWEEN contraintes.temp_opt_min
                            AND contraintes.temp_opt_max
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
                        BETWEEN contraintes.rayonnement_saison_min_kwh_m2
                                    / :rayonnement_divisor
                            AND contraintes.rayonnement_saison_max_kwh_m2
                                    / :rayonnement_divisor
                ) AS solar_ok,
                (
                    LOWER(TRIM(sols.soil_family_type)) IN (
                        LOWER(TRIM(contraintes.preferred_soil_family_1)),
                        LOWER(TRIM(contraintes.preferred_soil_family_2)),
                        LOWER(TRIM(contraintes.preferred_soil_family_3))
                    )
                ) AS soil_ok,
                (futur.nb_mois_future = 5) AS donnees_futures_completes
            FROM public.dim_coordinates AS coord
            JOIN public.dim_soils AS sols
              ON coord.coordinates_id = sols.coordinates_id
            LEFT JOIN climat_futur_saison AS futur
              ON coord.coordinates_id = futur.coordinates_id
            CROSS JOIN contraintes
            LEFT JOIN historique_2024 AS hist
              ON coord.coordinates_id = hist.coordinates_id
            LEFT JOIN climat_2024_saison AS passe
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
            soil_type,
            department_code,
            department_name,
            region_code,
            region_name,
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
            temp_ok,
            precip_ok,
            solar_ok,
            soil_ok,
            donnees_futures_completes,
            apte,
            raison_echec,
            CASE
                WHEN historique_2024 AND apte THEN 'verte'
                WHEN historique_2024 AND NOT apte THEN 'rouge'
                WHEN NOT historique_2024 AND apte THEN 'bleue'
                ELSE 'non_affichee'
            END AS statut
        FROM evaluation
        WHERE historique_2024 OR apte;
    """

    return read_dataframe(
        engine,
        query,
        {
            "projection_year": int(projection_year),
            "culture_id": int(culture_id),
            "rayonnement_divisor": float(rayonnement_divisor),
        },
    )


def train_yield_model(engine: Engine, culture_id: int) -> ModelBundle:
    df_historique = load_yield_training_data(engine, culture_id).copy()
    target = "rendement_q_ha"
    useful_columns = (
        FEATURES_RENDEMENT_NUMERIQUES
        + FEATURES_RENDEMENT_CATEGORIELLES
        + [target, "year", "coordinates_id", "rayonnement_saison_kwh_m2"]
    )

    missing = set(useful_columns) - set(df_historique.columns)
    if missing:
        raise ValueError(
            f"Colonnes d'entraînement manquantes pour la culture {culture_id}: "
            f"{sorted(missing)}"
        )

    df_model = df_historique[useful_columns].copy()
    df_model[target] = pd.to_numeric(df_model[target], errors="coerce")
    df_model = df_model.dropna(subset=[target])

    if len(df_model) < 50:
        raise ValueError(
            f"Culture {culture_id}: seulement {len(df_model)} lignes valides ; "
            "au moins 50 sont nécessaires."
        )

    rendement_min = df_model[target].quantile(0.01)
    rendement_max = df_model[target].quantile(0.99)
    df_model = df_model[
        df_model[target].between(rendement_min, rendement_max)
    ].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "variables_numeriques",
                Pipeline(
                    steps=[
                        ("imputation", SimpleImputer(strategy="median")),
                    ]
                ),
                FEATURES_RENDEMENT_NUMERIQUES,
            ),
            (
                "variables_categorielles",
                Pipeline(
                    steps=[
                        (
                            "imputation",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encodage",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                FEATURES_RENDEMENT_CATEGORIELLES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preparation", preprocessor),
            (
                "regression",
                RandomForestRegressor(
                    n_estimators=40,
                    max_depth=10,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model_variables = (
        FEATURES_RENDEMENT_NUMERIQUES
        + FEATURES_RENDEMENT_CATEGORIELLES
    )
    training = df_model[df_model["year"] <= 2022]
    validation = df_model[df_model["year"] >= 2023]

    metrics: dict[str, float | int | None] = {
        "mae": None,
        "r2": None,
        "nb_lignes": len(df_model),
    }

    if len(training) >= 50 and len(validation) >= 10:
        model.fit(training[model_variables], training[target])
        predictions = model.predict(validation[model_variables])
        metrics["mae"] = float(mean_absolute_error(validation[target], predictions))
        metrics["r2"] = float(r2_score(validation[target], predictions))

    model.fit(df_model[model_variables], df_model[target])

    rendement_reference_haut = float(df_model[target].quantile(0.90))
    reference_par_maille = (
        df_model.groupby("coordinates_id", as_index=False)
        .agg(
            rendement_reference=("rendement_q_ha", "mean"),
            temp_saison_reference=("temp_saison", "mean"),
            temp_max_reference=("temp_max_saison", "mean"),
            precip_cumul_reference=("precip_cumul", "mean"),
            solar_moy_reference=("solar_moy", "mean"),
            rayonnement_reference_kwh_m2=(
                "rayonnement_saison_kwh_m2",
                "mean",
            ),
        )
    )

    valeurs_reference_culture = {
        "rendement_reference": float(df_model["rendement_q_ha"].median()),
        "temp_saison_reference": float(df_model["temp_saison"].median()),
        "temp_max_reference": float(df_model["temp_max_saison"].median()),
        "precip_cumul_reference": float(df_model["precip_cumul"].median()),
        "solar_moy_reference": float(df_model["solar_moy"].median()),
        "rayonnement_reference_kwh_m2": float(
            df_model["rayonnement_saison_kwh_m2"].median()
        ),
    }

    return ModelBundle(
        model=model,
        rendement_reference_haut=rendement_reference_haut,
        reference_par_maille=reference_par_maille,
        valeurs_reference_culture=valeurs_reference_culture,
        metrics=metrics,
    )


def calculer_score_intervalle(
    valeur: Any,
    borne_optimale_min: Any,
    borne_optimale_max: Any,
    limite_absolue_min: Any = None,
    limite_absolue_max: Any = None,
) -> float:
    required = [valeur, borne_optimale_min, borne_optimale_max]
    if any(pd.isna(element) for element in required):
        return 0.0

    valeur = float(valeur)
    borne_optimale_min = float(borne_optimale_min)
    borne_optimale_max = float(borne_optimale_max)
    amplitude = max(borne_optimale_max - borne_optimale_min, 0.0001)

    if limite_absolue_min is None or pd.isna(limite_absolue_min):
        limite_absolue_min = max(0.0, borne_optimale_min - amplitude)
    if limite_absolue_max is None or pd.isna(limite_absolue_max):
        limite_absolue_max = borne_optimale_max + amplitude

    limite_absolue_min = float(limite_absolue_min)
    limite_absolue_max = float(limite_absolue_max)

    if borne_optimale_min <= valeur <= borne_optimale_max:
        return 100.0
    if valeur < borne_optimale_min:
        if valeur <= limite_absolue_min:
            return 0.0
        denominator = borne_optimale_min - limite_absolue_min
        if denominator <= 0:
            return 0.0
        return float(
            np.clip(
                100 * (valeur - limite_absolue_min) / denominator,
                0,
                100,
            )
        )
    if valeur >= limite_absolue_max:
        return 0.0

    denominator = limite_absolue_max - borne_optimale_max
    if denominator <= 0:
        return 0.0
    return float(
        np.clip(
            100 * (limite_absolue_max - valeur) / denominator,
            0,
            100,
        )
    )


def build_harvest_score_data(
    engine: Engine,
    culture_id: int,
    projection_year: int,
    constraints: pd.DataFrame,
    model_bundle: ModelBundle,
    coefficient_nouvelles_zones: float,
    rayonnement_divisor: float,
) -> pd.DataFrame:
    # La projection agronomique est la table pilote : elle contient aussi les
    # mailles historiques dépourvues de climat futur. Celles-ci doivent rester
    # dans la table de faits avec un statut rouge, et non disparaître lors
    # d'une jointure interne avec les variables du modèle.
    df_projection = load_crop_projection(
        engine,
        projection_year,
        culture_id,
        rayonnement_divisor,
    ).copy()
    if df_projection.empty:
        return df_projection

    df_future = load_future_yield_features(engine, projection_year).copy()

    projection_columns = [
        "coordinates_id",
        "lat",
        "lon",
        "soil_family_type",
        "soil_type",
        "department_code",
        "department_name",
        "region_code",
        "region_name",
        "statut",
        "historique_2024",
        "surface_ha_12_5km",
        "nb_parcelles_12_5km",
        "temp_moyenne_2024",
        "temp_max_2024",
        "precip_saison_2024",
        "rayonnement_saison_2024",
        "temp_moyenne_future",
        "temp_max_future",
        "precip_saison_future",
        "rayonnement_saison_future",
        "temp_ok",
        "precip_ok",
        "solar_ok",
        "soil_ok",
        "donnees_futures_completes",
        "apte",
        "raison_echec",
    ]
    df_projection = df_projection[projection_columns].drop_duplicates(
        subset=["coordinates_id"],
        keep="first",
    )

    # On ne récupère depuis df_future que les variables numériques du modèle.
    # Les coordonnées et le sol viennent de df_projection, y compris pour les
    # mailles sans données climatiques futures.
    future_feature_columns = [
        "coordinates_id",
        "temp_saison",
        "temp_max_saison",
        "precip_cumul",
        "solar_moy",
        "rayonnement_saison_kwh_m2",
    ]

    if df_future.empty:
        df = df_projection.copy()

        for column in future_feature_columns[1:]:
            df[column] = np.nan

    else:
        df = df_projection.merge(
            df_future[future_feature_columns],
            on="coordinates_id",
            how="left",
            validate="one_to_one",
        )


    # Les mailles sans données climatiques futures reçoivent False.
    df["donnees_futures_completes"] = (
        df["donnees_futures_completes"]
        .fillna(False)
        .astype(bool)
    )


    model_variables = (
        FEATURES_RENDEMENT_NUMERIQUES
        + FEATURES_RENDEMENT_CATEGORIELLES
    )

    df["donnees_futures_completes"] = (
        df["donnees_futures_completes"]
        .fillna(False)
        .astype(bool)
    )

    # Une prédiction n'est calculée que lorsque les cinq mois climatiques sont
    # disponibles. 

    prediction_mask = (
        df["donnees_futures_completes"]
        & df[FEATURES_RENDEMENT_NUMERIQUES].notna().all(axis=1)
        & df["soil_family_type"].notna()
    )

    df["rendement_potentiel_q_ha"] = 0.0
    if prediction_mask.any():
        df.loc[
            prediction_mask,
            "rendement_potentiel_q_ha",
        ] = model_bundle.model.predict(
            df.loc[prediction_mask, model_variables]
        )

    df["rendement_potentiel_q_ha"] = df[
        "rendement_potentiel_q_ha"
    ].clip(lower=0)

    df = df.merge(
        model_bundle.reference_par_maille,
        on="coordinates_id",
        how="left",
        validate="one_to_one",
    )

    for column, default_value in model_bundle.valeurs_reference_culture.items():
        df[column] = df[column].fillna(default_value)

    constraint_rows = constraints.loc[constraints["culture_id"] == culture_id]
    if constraint_rows.empty:
        raise ValueError(f"Aucune contrainte pour la culture {culture_id}.")
    constraint = constraint_rows.iloc[0]

    preferred_soils = {
        str(soil).strip().lower()
        for soil in [
            constraint["preferred_soil_family_1"],
            constraint["preferred_soil_family_2"],
            constraint["preferred_soil_family_3"],
        ]
        if pd.notna(soil)
    }
    df["score_sol"] = np.where(
        df["soil_family_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(preferred_soils),
        100.0,
        0.0,
    )

    rendement_reference_haut = max(
        float(model_bundle.rendement_reference_haut),
        0.0001,
    )
    df["score_rendement_reference"] = (
        df["rendement_reference"] / rendement_reference_haut * 100
    ).clip(0, 100)
    df["score_rendement_futur"] = (
        df["rendement_potentiel_q_ha"] / rendement_reference_haut * 100
    ).clip(0, 100)

    temp_min = (
        constraint["temp_saison_min"]
        if pd.notna(constraint["temp_saison_min"])
        else constraint["temp_opt_min"]
    )
    temp_max = (
        constraint["temp_saison_max"]
        if pd.notna(constraint["temp_saison_max"])
        else constraint["temp_opt_max"]
    )
    df["score_temperature_reference"] = df["temp_saison_reference"].apply(
        lambda value: calculer_score_intervalle(
            value,
            temp_min,
            temp_max,
            constraint["temp_abs_min"],
            constraint["temp_abs_max"],
        )
    )
    df["score_temperature_futur"] = df["temp_saison"].apply(
        lambda value: calculer_score_intervalle(
            value,
            temp_min,
            temp_max,
            constraint["temp_abs_min"],
            constraint["temp_abs_max"],
        )
    )

    precip_min = float(constraint["precip_saison_min"])
    precip_max = float(constraint["precip_saison_max"])
    precip_amplitude = max(precip_max - precip_min, 0.0001)
    precip_absolute_max = precip_max + precip_amplitude
    df["score_precip_reference"] = df["precip_cumul_reference"].apply(
        lambda value: calculer_score_intervalle(
            value,
            precip_min,
            precip_max,
            0,
            precip_absolute_max,
        )
    )
    df["score_precip_futur"] = df["precip_cumul"].apply(
        lambda value: calculer_score_intervalle(
            value,
            precip_min,
            precip_max,
            0,
            precip_absolute_max,
        )
    )

    ray_min = (
        float(constraint["rayonnement_saison_min_kwh_m2"])
        / rayonnement_divisor
    )
    ray_max = (
        float(constraint["rayonnement_saison_max_kwh_m2"])
        / rayonnement_divisor
    )
    ray_amplitude = max(ray_max - ray_min, 0.0001)
    df["score_rayonnement_reference"] = df[
        "rayonnement_reference_kwh_m2"
    ].apply(
        lambda value: calculer_score_intervalle(
            value,
            ray_min,
            ray_max,
            max(0, ray_min - ray_amplitude),
            ray_max + ray_amplitude,
        )
    )
    df["score_rayonnement_futur"] = df[
        "rayonnement_saison_kwh_m2"
    ].apply(
        lambda value: calculer_score_intervalle(
            value,
            ray_min,
            ray_max,
            max(0, ray_min - ray_amplitude),
            ray_max + ray_amplitude,
        )
    )

    df["harvest_score_reference"] = (
        df["score_rendement_reference"] * POIDS_HARVEST_SCORE["rendement"]
        + df["score_temperature_reference"]
        * POIDS_HARVEST_SCORE["temperature"]
        + df["score_precip_reference"]
        * POIDS_HARVEST_SCORE["precipitations"]
        + df["score_rayonnement_reference"]
        * POIDS_HARVEST_SCORE["rayonnement"]
        + df["score_sol"] * POIDS_HARVEST_SCORE["sol"]
    ).clip(0, 100).round(1)
    df["harvest_score_futur"] = (
        df["score_rendement_futur"] * POIDS_HARVEST_SCORE["rendement"]
        + df["score_temperature_futur"]
        * POIDS_HARVEST_SCORE["temperature"]
        + df["score_precip_futur"]
        * POIDS_HARVEST_SCORE["precipitations"]
        + df["score_rayonnement_futur"]
        * POIDS_HARVEST_SCORE["rayonnement"]
        + df["score_sol"] * POIDS_HARVEST_SCORE["sol"]
    ).clip(0, 100).round(1)

    # Sans climat futur complet, aucun score futur fiable ne peut être établi.
    # La maille reste rouge, avec une contribution productive future nulle.
    incomplete_future = ~df["donnees_futures_completes"].fillna(False).astype(bool)
    future_score_columns = [
        "score_rendement_futur",
        "score_temperature_futur",
        "score_precip_futur",
        "score_rayonnement_futur",
        "harvest_score_futur",
    ]
    df.loc[incomplete_future, future_score_columns] = 0.0

    df["evolution_score_points"] = (
        df["harvest_score_futur"] - df["harvest_score_reference"]
    ).round(1)
    df["evolution_rendement_pct"] = (
        (df["rendement_potentiel_q_ha"] - df["rendement_reference"])
        .div(df["rendement_reference"].replace(0, pd.NA))
        .mul(100)
        .round(1)
    )

    df["surface_ha_12_5km"] = pd.to_numeric(
        df["surface_ha_12_5km"],
        errors="coerce",
    ).fillna(0)
    df["department_code"] = df["department_code"].astype("string").str.strip()
    df["department_name"] = df["department_name"].astype("string").str.strip()

    historical_surfaces = df.loc[
        df["historique_2024"] & (df["surface_ha_12_5km"] > 0),
        "surface_ha_12_5km",
    ]
    national_median = (
        float(historical_surfaces.median())
        if not historical_surfaces.empty
        else 0.0
    )

    department_medians = (
        df.loc[
            df["historique_2024"]
            & (df["surface_ha_12_5km"] > 0)
            & df["department_code"].notna()
            & df["department_name"].notna()
        ]
        .groupby(["department_code", "department_name"], as_index=False)
        .agg(
            surface_mediane_historique_maille=(
                "surface_ha_12_5km",
                "median",
            )
        )
    )
    df = df.merge(
        department_medians,
        on=["department_code", "department_name"],
        how="left",
        validate="many_to_one",
    )
    df["surface_mediane_historique_maille"] = df[
        "surface_mediane_historique_maille"
    ].fillna(national_median)

    df["surface_future_estimee_ha"] = np.select(
        [
            df["statut"].eq("verte"),
            df["statut"].eq("bleue"),
            df["statut"].eq("rouge"),
        ],
        [
            df["surface_ha_12_5km"],
            df["surface_mediane_historique_maille"]
            * float(coefficient_nouvelles_zones),
            0.0,
        ],
        default=0.0,
    )
    df["surface_reference_2024_ha"] = np.where(
        df["historique_2024"],
        df["surface_ha_12_5km"],
        0.0,
    )
    df["surface_nouvelle_estimee_ha"] = np.where(
        df["statut"].eq("bleue"),
        df["surface_future_estimee_ha"],
        0.0,
    )

    df["culture_id"] = int(culture_id)
    df["projection_year"] = int(projection_year)
    df["model_version"] = MODEL_VERSION
    df["coefficient_nouvelles_zones"] = float(
        coefficient_nouvelles_zones
    )
    df["nb_parcelles_2024"] = df["nb_parcelles_12_5km"]
    df["rendement_reference_q_ha"] = df["rendement_reference"]
    df["rayonnement_saison_2024_kwh_m2"] = df[
        "rayonnement_saison_2024"
    ]
    df["rayonnement_saison_future_kwh_m2"] = df[
        "rayonnement_saison_future"
    ]
    df["surface_mediane_historique_maille_ha"] = df[
        "surface_mediane_historique_maille"
    ]

    missing_fact_columns = set(FACT_COLUMNS) - set(df.columns)
    if missing_fact_columns:
        raise ValueError(
            "Colonnes manquantes avant insertion dans la table de faits : "
            f"{sorted(missing_fact_columns)}"
        )

    return df[FACT_COLUMNS].copy()


def _to_python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _to_python_value(value) for column, value in row.items()}
        for row in dataframe.to_dict(orient="records")
    ]


def chunked(
    values: Sequence[dict[str, Any]],
    size: int,
) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def replace_fact_partition(
    engine: Engine,
    dataframe: pd.DataFrame,
    culture_id: int,
    projection_year: int,
) -> int:
    """Remplace atomiquement une partition culture × année."""

    records = dataframe_to_records(dataframe)
    if not records:
        raise ValueError(
            f"Aucune ligne à charger pour culture={culture_id}, "
            f"année={projection_year}."
        )

    delete_query = text("""
        DELETE FROM public.fact_crop_projection
        WHERE culture_id = :culture_id
          AND projection_year = :projection_year;
    """)

    placeholders = ",\n            ".join(f":{column}" for column in FACT_COLUMNS)
    column_sql = ",\n            ".join(FACT_COLUMNS)
    insert_query = text(
        f"""
        INSERT INTO public.fact_crop_projection (
            {column_sql}
        )
        VALUES (
            {placeholders}
        );
        """
    )

    with engine.begin() as connection:
        connection.execute(
            delete_query,
            {
                "culture_id": int(culture_id),
                "projection_year": int(projection_year),
            },
        )
        for batch in chunked(records, 1000):
            connection.execute(insert_query, list(batch))

    return len(records)


def validate_fact_dataframe(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        raise ValueError("Le DataFrame final est vide.")

    duplicated = dataframe.duplicated(
        subset=["coordinates_id", "culture_id", "projection_year"],
        keep=False,
    )
    if duplicated.any():
        examples = dataframe.loc[
            duplicated,
            ["coordinates_id", "culture_id", "projection_year"],
        ].head(10)
        raise ValueError(
            "Clés métier dupliquées avant insertion :\n"
            f"{examples.to_string(index=False)}"
        )

    invalid_statuses = set(dataframe["statut"].dropna().unique()) - {
        "verte",
        "rouge",
        "bleue",
        "non_affichee",
    }
    if invalid_statuses:
        raise ValueError(f"Statuts invalides : {sorted(invalid_statuses)}")

    for column in [
        "surface_reference_2024_ha",
        "surface_future_estimee_ha",
        "surface_nouvelle_estimee_ha",
    ]:
        if (dataframe[column] < 0).any():
            raise ValueError(f"La colonne {column} contient une valeur négative.")

    scores = dataframe["harvest_score_futur"].dropna()
    if not scores.between(0, 100).all():
        raise ValueError("Un Harvest Score futur est hors de l'intervalle 0–100.")

    incomplete_historical = dataframe[
        dataframe["historique_2024"]
        & ~dataframe["donnees_futures_completes"].fillna(False).astype(bool)
    ]
    if not incomplete_historical.empty:
        invalid_incomplete_status = incomplete_historical[
            ~incomplete_historical["statut"].eq("rouge")
        ]
        if not invalid_incomplete_status.empty:
            raise ValueError(
                "Une maille historique sans climat futur complet n'est pas rouge."
            )


def summarize_partition(dataframe: pd.DataFrame) -> str:
    status_counts = dataframe["statut"].value_counts().to_dict()
    reference_surface = dataframe["surface_reference_2024_ha"].sum()
    future_surface = dataframe["surface_future_estimee_ha"].sum()
    return (
        f"{len(dataframe):,} mailles | "
        f"verte={status_counts.get('verte', 0):,}, "
        f"rouge={status_counts.get('rouge', 0):,}, "
        f"bleue={status_counts.get('bleue', 0):,} | "
        f"surface 2024={reference_surface:,.1f} ha | "
        f"surface future={future_surface:,.1f} ha"
    ).replace(",", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construit fact_crop_projection dans Neon."
    )
    parser.add_argument(
        "--culture-id",
        type=int,
        choices=CULTURE_IDS,
        help="Ne traiter qu'une culture.",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=PROJECTION_YEARS,
        help="Ne traiter qu'une année de projection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcule et valide les résultats sans écrire dans Neon.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    database_url = get_database_url()
    rayonnement_divisor = get_rayonnement_divisor()
    engine = make_engine(database_url)

    cultures = [args.culture_id] if args.culture_id else list(CULTURE_IDS)
    years = [args.year] if args.year else list(PROJECTION_YEARS)
    constraints = load_constraints(engine)

    expected_cultures = set(cultures)
    available_cultures = set(
        pd.to_numeric(constraints["culture_id"], errors="coerce")
        .dropna()
        .astype(int)
    )
    missing_cultures = expected_cultures - available_cultures
    if missing_cultures:
        raise RuntimeError(
            "Cultures absentes de dim_constraints : "
            f"{sorted(missing_cultures)}"
        )

    failures: list[str] = []

    for culture_id in cultures:
        logging.info("Entraînement du modèle pour la culture %s", culture_id)
        try:
            model_bundle = train_yield_model(engine, culture_id)
        except Exception as exc:  # noqa: BLE001 - journalisation globale de l'ETL
            logging.exception(
                "Échec de l'entraînement pour la culture %s",
                culture_id,
            )
            failures.append(f"culture {culture_id}: {exc}")
            continue

        logging.info(
            "Modèle culture %s | lignes=%s | MAE=%s | R²=%s",
            culture_id,
            model_bundle.metrics["nb_lignes"],
            model_bundle.metrics["mae"],
            model_bundle.metrics["r2"],
        )

        for projection_year in years:
            logging.info(
                "Calcul culture=%s, année=%s",
                culture_id,
                projection_year,
            )
            try:
                dataframe = build_harvest_score_data(
                    engine=engine,
                    culture_id=culture_id,
                    projection_year=projection_year,
                    constraints=constraints,
                    model_bundle=model_bundle,
                    coefficient_nouvelles_zones=COEFFICIENT_NOUVELLES_ZONES,
                    rayonnement_divisor=rayonnement_divisor,
                )
                validate_fact_dataframe(dataframe)
                logging.info(summarize_partition(dataframe))

                if args.dry_run:
                    logging.info(
                        "DRY RUN : aucune écriture pour culture=%s, année=%s",
                        culture_id,
                        projection_year,
                    )
                else:
                    inserted = replace_fact_partition(
                        engine,
                        dataframe,
                        culture_id,
                        projection_year,
                    )
                    logging.info(
                        "%s lignes chargées dans fact_crop_projection.",
                        inserted,
                    )
            except Exception as exc:  # noqa: BLE001
                logging.exception(
                    "Échec culture=%s, année=%s",
                    culture_id,
                    projection_year,
                )
                failures.append(
                    f"culture {culture_id}, année {projection_year}: {exc}"
                )

    if failures:
        raise RuntimeError(
            "L'ETL s'est terminé avec des erreurs :\n- "
            + "\n- ".join(failures)
        )

    logging.info("ETL terminé avec succès.")


if __name__ == "__main__":
    main()
