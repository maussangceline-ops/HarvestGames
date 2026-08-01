import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIGURATION GÉNÉRALE
# =========================================================

st.set_page_config(
    page_title="Harvest Games",
    page_icon="🌾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

CULTURE_INFO = {
    "Vigne à raisin de cuve": {"picto": "🍇", "id": 1},
    "Pois chiche": {"picto": "🌱", "id": 2},
    "Pomme de terre": {"picto": "🥔", "id": 3},
}

ANNEES_PROJECTION = [2054, 2084, 2100]

POIDS_HARVEST_SCORE = {
    "Rendement potentiel": 15,
    "Température": 35,
    "Précipitations": 20,
    "Rayonnement": 15,
    "Sol": 15,
}

COULEURS_STATUT = {
    "verte": "#2ecc71",
    "rouge": "#e74c3c",
    "bleue": "#3498db",
}


# =========================================================
# CONNEXION NEON
# =========================================================


def get_database_url() -> str:
    """Lit l'URL Neon depuis Docker ou depuis les secrets Streamlit."""
    database_url = os.getenv("NEON_DATABASE_URL")

    if database_url:
        return database_url

    try:
        database_url = st.secrets["NEON_DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        database_url = None

    if not database_url:
        raise RuntimeError(
            "NEON_DATABASE_URL est absente. Ajoute-la dans le fichier .env "
            "pour Docker ou dans .streamlit/secrets.toml pour Streamlit local."
        )

    return str(database_url)


conn = st.connection(
    "postgresql",
    type="sql",
    url=get_database_url(),
    pool_recycle=300,
    connect_args={
        "sslmode": "require",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)


# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================


def format_entier(value) -> str:
    if pd.isna(value):
        return "Non disponible"
    return f"{float(value):,.0f}".replace(",", " ")


def format_decimal(value, digits: int = 1) -> str:
    if pd.isna(value):
        return "Non disponible"
    return f"{float(value):,.{digits}f}".replace(",", " ")


def format_hover_value(value, digits: int = 1, suffix: str = "") -> str:
    if pd.isna(value):
        return "non disponible"
    return f"{float(value):,.{digits}f}{suffix}".replace(",", " ")


def normaliser_booleens(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].fillna(False).astype(bool)
    return dataframe


def oui_non(value: bool) -> str:
    return "✅" if bool(value) else "❌"


# =========================================================
# CHARGEMENTS NEON
# =========================================================


@st.cache_data(ttl=600)
def load_constraints() -> pd.DataFrame:
    return conn.query(
        """
        SELECT *
        FROM public.dim_constraints
        ORDER BY culture_id;
        """,
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_national_stats(culture_id: int) -> pd.DataFrame:
    """Statistiques 2024 issues de la vue nettoyée des parcelles."""
    return conn.query(
        """
        SELECT
            COUNT(DISTINCT parcel_id) AS nb_parcelles,
            COUNT(DISTINCT coordinates_id) AS nb_mailles,
            SUM(surf_ha) AS total_ha
        FROM public.vw_cultures_plots_2024_clean
        WHERE culture_id = :culture_id;
        """,
        params={"culture_id": int(culture_id)},
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_current_culture_data(
    culture_id: int,
    annee_cible: int = 2054,
) -> pd.DataFrame:
    """Données mensuelles utilisées dans la carte thermique."""
    return conn.query(
        """
        WITH cultures_par_maille AS (
            SELECT
                coordinates_id,
                COUNT(DISTINCT parcel_id) AS nb_parcelles,
                SUM(surf_ha) AS surface_ha_exacte
            FROM public.vw_cultures_plots_2024_clean
            WHERE culture_id = :culture_id
            GROUP BY coordinates_id
        ),

        climat_2024 AS (
            SELECT
                coordinates_id,
                month,
                AVG(temp_mean_past) AS temp_2024
            FROM public.src_climate_past
            WHERE year = 2024
              AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, month
        ),

        climat_futur AS (
            SELECT
                coordinates_id,
                month,
                AVG(temp_mean_future) AS temp_future
            FROM public.src_climate_future
            WHERE year = :annee_cible
              AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, month
        )

        SELECT
            cultures.coordinates_id,
            coord.lat,
            coord.lon,
            futur.month,
            passe.temp_2024,
            futur.temp_future,
            cultures.nb_parcelles,
            cultures.surface_ha_exacte
        FROM cultures_par_maille AS cultures
        JOIN public.dim_coordinates AS coord
          ON coord.coordinates_id = cultures.coordinates_id
        JOIN climat_futur AS futur
          ON futur.coordinates_id = cultures.coordinates_id
        LEFT JOIN climat_2024 AS passe
          ON passe.coordinates_id = cultures.coordinates_id
         AND passe.month = futur.month
        ORDER BY cultures.coordinates_id, futur.month;
        """,
        params={
            "culture_id": int(culture_id),
            "annee_cible": int(annee_cible),
        },
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_crop_projection(annee_sel: int, culture_id: int) -> pd.DataFrame:
    """Lit les projections déjà calculées par l'ETL dans Neon."""
    return conn.query(
        """
        SELECT
            fact.projection_id,
            fact.coordinates_id,
            fact.culture_id,
            fact.projection_year,

            fact.lat,
            fact.lon,
            fact.department_code,
            fact.department_name,
            fact.region_code,
            fact.region_name,
            fact.soil_family_type,
            fact.soil_type,

            fact.historique_2024,
            fact.nb_parcelles_2024 AS nb_parcelles_12_5km,
            fact.surface_reference_2024_ha AS surface_ha_12_5km,
            fact.surface_future_estimee_ha,
            fact.surface_nouvelle_estimee_ha,

            fact.temp_moyenne_2024,
            fact.temp_max_2024,
            fact.precip_saison_2024,
            fact.rayonnement_saison_2024_kwh_m2
                AS rayonnement_saison_2024,

            fact.temp_moyenne_future,
            fact.temp_max_future,
            fact.precip_saison_future,
            fact.rayonnement_saison_future_kwh_m2
                AS rayonnement_saison_future,

            contraintes.temp_opt_min,
            contraintes.temp_opt_max,
            contraintes.precip_saison_min,
            contraintes.precip_saison_max,
            contraintes.rayonnement_saison_min_kwh_m2 / 3.6
                AS rayonnement_saison_min_kwh_m2,
            contraintes.rayonnement_saison_max_kwh_m2 / 3.6
                AS rayonnement_saison_max_kwh_m2,

            fact.temp_ok,
            fact.precip_ok,
            fact.solar_ok,
            fact.soil_ok,
            fact.donnees_futures_completes,
            fact.apte,
            fact.statut,
            fact.raison_echec,

            fact.rendement_reference_q_ha,
            fact.rendement_potentiel_q_ha,
            fact.score_rendement_reference,
            fact.score_rendement_futur,
            fact.score_temperature_reference,
            fact.score_temperature_futur,
            fact.score_precip_reference,
            fact.score_precip_futur,
            fact.score_rayonnement_reference,
            fact.score_rayonnement_futur,
            fact.score_sol,
            fact.harvest_score_reference,
            fact.harvest_score_futur,
            fact.evolution_score_points,
            fact.evolution_rendement_pct,
            fact.model_version,
            fact.calculated_at

        FROM public.fact_crop_projection AS fact
        JOIN public.dim_constraints AS contraintes
          ON contraintes.culture_id = fact.culture_id

        WHERE fact.culture_id = :culture_id
          AND fact.projection_year = :annee_sel
          AND fact.statut <> 'non_affichee'

        ORDER BY fact.coordinates_id;
        """,
        params={
            "culture_id": int(culture_id),
            "annee_sel": int(annee_sel),
        },
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_national_kpis(culture_id: int, projection_year: int) -> pd.DataFrame:
    """KPIs nationaux calculés par PostgreSQL à partir de la table de faits."""
    return conn.query(
        """
        WITH agregats AS (
            SELECT
                culture_id,
                projection_year,

                COUNT(*) AS nombre_mailles,
                COUNT(*) FILTER (WHERE historique_2024)
                    AS mailles_historiques,
                COUNT(*) FILTER (WHERE statut = 'verte') AS mailles_vertes,
                COUNT(*) FILTER (WHERE statut = 'rouge') AS mailles_rouges,
                COUNT(*) FILTER (WHERE statut = 'bleue') AS mailles_bleues,
                COUNT(*) FILTER (
                    WHERE donnees_futures_completes = FALSE
                ) AS mailles_sans_climat,

                SUM(surface_reference_2024_ha) AS surface_reference_2024_ha,
                SUM(surface_future_estimee_ha) FILTER (
                    WHERE statut = 'verte'
                ) AS surface_maintenue_ha,
                SUM(surface_reference_2024_ha) FILTER (
                    WHERE statut = 'rouge'
                ) AS surface_perdue_ha,
                SUM(surface_nouvelle_estimee_ha)
                    AS surface_nouvelle_estimee_ha,
                SUM(surface_future_estimee_ha) AS surface_future_estimee_ha,

                SUM(
                    COALESCE(harvest_score_futur, 0)
                    * surface_future_estimee_ha
                ) / NULLIF(
                    SUM(surface_reference_2024_ha)
                    + SUM(surface_nouvelle_estimee_ha),
                    0
                ) AS harvest_score_national,

                SUM(
                    rendement_potentiel_q_ha
                    * surface_future_estimee_ha
                ) / NULLIF(
                    SUM(surface_future_estimee_ha),
                    0
                ) AS rendement_national_futur_q_ha,

                SUM(
                    rendement_reference_q_ha
                    * surface_reference_2024_ha
                ) / NULLIF(
                    SUM(surface_reference_2024_ha),
                    0
                ) AS rendement_reference_national_q_ha,

                SUM(
                    rendement_reference_q_ha
                    * surface_reference_2024_ha
                ) AS production_reference_2024_q,

                SUM(
                    rendement_potentiel_q_ha
                    * surface_future_estimee_ha
                ) AS production_future_estimee_q,

                MAX(calculated_at) AS calculated_at

            FROM public.fact_crop_projection
            WHERE culture_id = :culture_id
              AND projection_year = :projection_year
            GROUP BY culture_id, projection_year
        )

        SELECT
            *,
            (
                surface_future_estimee_ha - surface_reference_2024_ha
            ) / NULLIF(surface_reference_2024_ha, 0) * 100
                AS evolution_surface_pct,

            (
                rendement_national_futur_q_ha
                - rendement_reference_national_q_ha
            ) / NULLIF(rendement_reference_national_q_ha, 0) * 100
                AS evolution_rendement_pct,

            (
                production_future_estimee_q
                - production_reference_2024_q
            ) / NULLIF(production_reference_2024_q, 0) * 100
                AS evolution_production_pct

        FROM agregats;
        """,
        params={
            "culture_id": int(culture_id),
            "projection_year": int(projection_year),
        },
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_department_kpis(culture_id: int, projection_year: int) -> pd.DataFrame:
    """Agrégation départementale calculée directement dans Neon."""
    return conn.query(
        """
        WITH agregats AS (
            SELECT
                culture_id,
                projection_year,
                department_code,
                department_name,

                COUNT(*) FILTER (WHERE historique_2024)
                    AS nb_mailles_2024,
                COUNT(*) FILTER (WHERE statut = 'verte')
                    AS nb_mailles_maintenues,
                COUNT(*) FILTER (WHERE statut = 'rouge')
                    AS nb_mailles_perdues,
                COUNT(*) FILTER (WHERE statut = 'bleue')
                    AS nb_nouvelles_mailles_aptes,

                SUM(surface_reference_2024_ha)
                    AS surface_2024_ha,
                SUM(surface_future_estimee_ha) FILTER (
                    WHERE statut = 'verte'
                ) AS surface_maintenue_ha,
                SUM(surface_reference_2024_ha) FILTER (
                    WHERE statut = 'rouge'
                ) AS surface_perdue_ha,
                SUM(surface_nouvelle_estimee_ha)
                    AS surface_nouvelle_estimee_ha,
                SUM(surface_future_estimee_ha)
                    AS surface_future_estimee_ha,

                SUM(
                    rendement_potentiel_q_ha
                    * surface_future_estimee_ha
                ) / NULLIF(
                    SUM(surface_future_estimee_ha),
                    0
                ) AS rendement_potentiel_q_ha,

                SUM(
                    rendement_reference_q_ha
                    * surface_reference_2024_ha
                ) / NULLIF(
                    SUM(surface_reference_2024_ha),
                    0
                ) AS rendement_reference_q_ha,

                SUM(
                    harvest_score_futur
                    * surface_future_estimee_ha
                ) / NULLIF(
                    SUM(surface_future_estimee_ha),
                    0
                ) AS harvest_score

            FROM public.fact_crop_projection
            WHERE culture_id = :culture_id
              AND projection_year = :projection_year
              AND department_code IS NOT NULL
              AND department_name IS NOT NULL
            GROUP BY
                culture_id,
                projection_year,
                department_code,
                department_name
        ),

        indicateurs AS (
            SELECT
                *,
                surface_future_estimee_ha - surface_2024_ha
                    AS evolution_surface_ha,
                (
                    surface_future_estimee_ha - surface_2024_ha
                ) / NULLIF(surface_2024_ha, 0) * 100
                    AS evolution_surface_pct,
                (
                    rendement_potentiel_q_ha - rendement_reference_q_ha
                ) / NULLIF(rendement_reference_q_ha, 0) * 100
                    AS evolution_rendement_pct,
                CASE
                    WHEN surface_2024_ha > 0 THEN TRUE
                    ELSE FALSE
                END AS a_reference_2024,
                COALESCE(
                    surface_maintenue_ha / NULLIF(surface_2024_ha, 0),
                    0
                ) AS taux_maintien,
                LEAST(
                    COALESCE(
                        surface_nouvelle_estimee_ha
                        / NULLIF(surface_2024_ha, 0),
                        0
                    ),
                    0.25
                ) AS bonus_nouvelles
            FROM agregats
        ),

        scores AS (
            SELECT
                *,
                GREATEST(
                    -100,
                    LEAST(
                        100,
                        ROUND(
                            taux_maintien * 100
                            + bonus_nouvelles * 100
                            - 50
                        )
                    )
                )::integer AS indice
            FROM indicateurs
        )

        SELECT
            *,
            CASE
                WHEN indice >= 80 THEN 'Très favorable'
                WHEN indice >= 60 THEN 'En progression'
                WHEN indice >= 40 THEN 'Plutôt favorable'
                WHEN indice >= 20 THEN 'Stable'
                WHEN indice >= 0 THEN 'Sous surveillance'
                ELSE 'Fortement vulnérable'
            END AS statut_departement,
            CASE
                WHEN indice >= 80 THEN '⭐⭐⭐⭐⭐'
                WHEN indice >= 60 THEN '⭐⭐⭐⭐☆'
                WHEN indice >= 40 THEN '⭐⭐⭐☆☆'
                WHEN indice >= 20 THEN '⭐⭐☆☆☆'
                WHEN indice >= 0 THEN '⭐☆☆☆☆'
                ELSE '☆☆☆☆☆'
            END AS etoiles
        FROM scores
        ORDER BY rendement_potentiel_q_ha DESC NULLS LAST;
        """,
        params={
            "culture_id": int(culture_id),
            "projection_year": int(projection_year),
        },
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_national_temperature_evolution() -> pd.DataFrame:
    return conn.query(
        """
        WITH past_by_cell AS (
            SELECT
                coordinates_id,
                AVG(temp_mean_past) AS temperature
            FROM public.src_climate_past
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
            FROM public.src_climate_future
            WHERE year IN (2054, 2084, 2100)
              AND month BETWEEN 4 AND 8
              AND temp_mean_future IS NOT NULL
            GROUP BY year, coordinates_id
        )

        SELECT 2024 AS year, AVG(temperature) AS temperature
        FROM past_by_cell

        UNION ALL

        SELECT year, AVG(temperature) AS temperature
        FROM future_by_cell
        GROUP BY year

        ORDER BY year;
        """,
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_local_temperature_evolution(coordinates_id: int) -> pd.DataFrame:
    return conn.query(
        """
        SELECT year, temperature
        FROM (
            SELECT
                2024 AS year,
                AVG(temp_mean_past) AS temperature
            FROM public.src_climate_past
            WHERE year = 2024
              AND month BETWEEN 4 AND 8
              AND coordinates_id = :coordinates_id

            UNION ALL

            SELECT
                year,
                AVG(temp_mean_future) AS temperature
            FROM public.src_climate_future
            WHERE year IN (2054, 2084, 2100)
              AND month BETWEEN 4 AND 8
              AND coordinates_id = :coordinates_id
            GROUP BY year
        ) AS evolution
        ORDER BY year;
        """,
        params={"coordinates_id": int(coordinates_id)},
        ttl=600,
    )


@st.cache_data(ttl=3600)
def load_departments_geojson() -> dict:
    geojson_path = ASSETS_DIR / "departements-1000m.geojson"
    with geojson_path.open("r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# DONNÉES COMMUNES
# =========================================================

try:
    df_constraints = load_constraints()
except Exception as error:
    st.error("La connexion à Neon a échoué.")
    st.exception(error)
    st.stop()


# =========================================================
# EN-TÊTE
# =========================================================

st.title("🌾 Harvest Games — Observatoire agricole du futur")

banner_path = ASSETS_DIR / "banner.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)

st.markdown(
    """
    <p style="font-size:22px;">
    Analysez l'impact du changement climatique sur la viabilité des cultures
    agricoles en France. Les cartes s'appuient sur les données agricoles 2024,
    les projections CORDEX et les résultats précalculés dans Neon.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

header_left, header_culture, header_parcelles, header_surface = st.columns(
    [1, 2, 2, 2]
)

with header_culture:
    culture_sel_header = st.selectbox(
        "Culture de référence :",
        list(CULTURE_INFO.keys()),
        key="culture_sel_header",
        format_func=lambda value: (
            f"{CULTURE_INFO[value]['picto']} {value}"
        ),
    )

culture_id_header = CULTURE_INFO[culture_sel_header]["id"]
stats_nat = load_national_stats(culture_id_header)

if stats_nat.empty:
    nb_parcelles = 0
    total_ha_france = 0
else:
    nb_parcelles = stats_nat.iloc[0]["nb_parcelles"]
    total_ha_france = stats_nat.iloc[0]["total_ha"]

with header_parcelles:
    st.metric(
        "Parcelles agricoles 2024",
        format_entier(nb_parcelles),
    )

with header_surface:
    st.metric(
        "Surface totale en France en 2024",
        f"{format_entier(total_ha_france)} ha",
    )


# =========================================================
# CARTE 1 — SEUILS THERMIQUES
# =========================================================

st.markdown("---")

col_cmd_1, col_carte_1 = st.columns([1, 3], gap="medium")

with col_cmd_1:
    st.markdown("### 🎛️ Paramètres")

    culture_sel_carte_1 = st.selectbox(
        "Sélectionnez la culture :",
        list(CULTURE_INFO.keys()),
        key="culture_sel_carte_1",
        format_func=lambda value: (
            f"{CULTURE_INFO[value]['picto']} {value}"
        ),
    )

    mode_analyse = st.radio(
        "Mode d'analyse",
        [
            "Situation observée (2024)",
            "Projection CORDEX 2054",
            "Simulation de hausse",
        ],
        key="mode_analyse_carte_1",
    )

    delta_temp = 0.0
    if mode_analyse == "Simulation de hausse":
        delta_temp = st.slider(
            "Hausse supplémentaire de température",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            key="delta_temperature_carte_1",
        )

culture_id_carte_1 = CULTURE_INFO[culture_sel_carte_1]["id"]
contrainte_carte_1 = df_constraints.loc[
    df_constraints["culture_id"] == culture_id_carte_1
]

if (
    not contrainte_carte_1.empty
    and pd.notna(contrainte_carte_1.iloc[0]["temp_opt_max"])
):
    seuil_alerte = float(contrainte_carte_1.iloc[0]["temp_opt_max"])
else:
    seuil_alerte = 30.0

with col_carte_1:
    st.subheader(
        "🌡️ Zones cultivées en 2024 menacées par la hausse des températures"
    )

    st.caption(
        "La carte conserve, pour chaque maille cultivée, le mois le plus "
        "défavorable entre avril et août."
    )

    try:
        df_current = load_current_culture_data(
            culture_id=culture_id_carte_1,
            annee_cible=2054,
        ).copy()
    except Exception as error:
        st.error("Les données de la carte thermique n'ont pas pu être chargées.")
        st.exception(error)
        df_current = pd.DataFrame()

    if df_current.empty:
        st.warning("Aucune donnée thermique disponible pour cette culture.")
    else:
        marge_orange = 2.0

        if mode_analyse == "Situation observée (2024)":
            df_current["temperature_analysee"] = df_current["temp_2024"]
            titre_carte_1 = (
                f"Situation observée en 2024 — {culture_sel_carte_1}"
            )
        elif mode_analyse == "Projection CORDEX 2054":
            df_current["temperature_analysee"] = df_current["temp_future"]
            titre_carte_1 = (
                f"Projection CORDEX 2054 — {culture_sel_carte_1}"
            )
        else:
            df_current["temperature_analysee"] = (
                df_current["temp_future"] + delta_temp
            )
            titre_carte_1 = (
                f"Projection 2054 + {delta_temp:.1f} °C — "
                f"{culture_sel_carte_1}"
            )

        def definir_statut_thermique(temperature):
            if pd.isna(temperature):
                return "Données manquantes"
            if temperature <= seuil_alerte:
                return "Viable"
            if temperature <= seuil_alerte + marge_orange:
                return "Critique"
            return "Fatale"

        df_current["statut_mensuel"] = df_current[
            "temperature_analysee"
        ].apply(definir_statut_thermique)

        ordre_statut = {
            "Données manquantes": 0,
            "Viable": 1,
            "Critique": 2,
            "Fatale": 3,
        }
        df_current["ordre"] = df_current["statut_mensuel"].map(ordre_statut)

        noms_mois = {
            4: "Avril",
            5: "Mai",
            6: "Juin",
            7: "Juillet",
            8: "Août",
        }
        df_current["mois"] = df_current["month"].map(noms_mois)

        df_carte_1 = (
            df_current.sort_values(
                ["coordinates_id", "ordre", "temperature_analysee"],
                ascending=[True, False, False],
            )
            .drop_duplicates("coordinates_id", keep="first")
            .rename(
                columns={
                    "statut_mensuel": "statut_parcelle",
                    "mois": "mois_defavorable",
                }
            )
        )

        total_surface = df_carte_1["surface_ha_exacte"].sum()

        surfaces_statut = (
            df_carte_1.groupby("statut_parcelle")["surface_ha_exacte"]
            .sum()
            .to_dict()
        )

        surf_viable = surfaces_statut.get("Viable", 0)
        surf_critique = surfaces_statut.get("Critique", 0)
        surf_fatale = surfaces_statut.get("Fatale", 0)

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric(
            "Surface analysée",
            f"{format_entier(total_surface)} ha",
        )
        metric_2.metric(
            "Zones viables 🟢",
            f"{surf_viable / total_surface * 100:.2f} %"
            if total_surface
            else "0 %",
        )
        metric_3.metric(
            "Zones critiques 🟠",
            f"{surf_critique / total_surface * 100:.2f} %"
            if total_surface
            else "0 %",
        )
        metric_4.metric(
            "Zones fatales 🔴",
            f"{surf_fatale / total_surface * 100:.2f} %"
            if total_surface
            else "0 %",
        )

        fig_carte_1 = px.scatter_mapbox(
            df_carte_1,
            lat="lat",
            lon="lon",
            color="statut_parcelle",
            size="surface_ha_exacte",
            size_max=12,
            color_discrete_map={
                "Viable": "#77dd77",
                "Critique": "#f39c12",
                "Fatale": "#e74c3c",
                "Données manquantes": "#7f8c8d",
            },
            category_orders={
                "statut_parcelle": [
                    "Viable",
                    "Critique",
                    "Fatale",
                    "Données manquantes",
                ]
            },
            mapbox_style=(
                "https://basemaps.cartocdn.com/gl/"
                "positron-nolabels-gl-style/style.json"
            ),
            zoom=4.3,
            center={"lat": 46.2, "lon": 2.2},
            title=titre_carte_1,
            opacity=0.8,
            hover_data={
                "statut_parcelle": True,
                "temperature_analysee": ":.1f",
                "mois_defavorable": True,
                "surface_ha_exacte": ":,.2f",
                "nb_parcelles": True,
                "lat": False,
                "lon": False,
                "coordinates_id": False,
            },
            labels={
                "statut_parcelle": "Statut",
                "temperature_analysee": "Température (°C)",
                "mois_defavorable": "Mois le plus défavorable",
                "surface_ha_exacte": "Surface cultivée (ha)",
                "nb_parcelles": "Parcelles",
            },
        )

        fig_carte_1.update_layout(
            height=600,
            margin={"r": 0, "t": 45, "l": 0, "b": 70},
            hoverlabel={"font_size": 18},
            legend={
                "title": None,
                "orientation": "h",
                "yanchor": "top",
                "y": -0.08,
                "xanchor": "center",
                "x": 0.5,
            },
        )

        st.plotly_chart(fig_carte_1, use_container_width=True)
        st.markdown(
            f"**Légende :** 🟢 jusqu'à {seuil_alerte:.1f} °C | "
            f"🟠 de {seuil_alerte:.1f} à "
            f"{seuil_alerte + marge_orange:.1f} °C | "
            f"🔴 au-delà de {seuil_alerte + marge_orange:.1f} °C"
        )


# =========================================================
# CARTE 2 — PROJECTIONS MULTICRITÈRES
# =========================================================

st.markdown("---")

col_param_2, col_carte_2 = st.columns([1, 3], gap="medium")

with col_param_2:
    st.markdown("### 🎛️ Paramètres")

    culture_sel_carte_2 = st.selectbox(
        "Sélectionnez la culture :",
        list(CULTURE_INFO.keys()),
        key="culture_sel_carte_2",
        format_func=lambda value: (
            f"{CULTURE_INFO[value]['picto']} {value}"
        ),
    )

    annee_sel_carte_2 = st.radio(
        "Scénario de projection",
        ANNEES_PROJECTION,
        horizontal=True,
        key="annee_sel_carte_2",
    )

culture_id_carte_2 = CULTURE_INFO[culture_sel_carte_2]["id"]

with col_carte_2:
    st.subheader("🗺️ Cartographie des opportunités")
    st.caption(
        "Les résultats sont lus directement dans fact_crop_projection. "
        "Aucun modèle n'est réentraîné lors de la consultation."
    )

    try:
        df_projection = load_crop_projection(
            annee_sel=annee_sel_carte_2,
            culture_id=culture_id_carte_2,
        )
    except Exception as error:
        st.error("La projection n'a pas pu être chargée depuis Neon.")
        st.exception(error)
        df_projection = pd.DataFrame()

    if df_projection.empty:
        st.warning("Aucune projection disponible pour ce scénario.")
    else:
        df_projection = normaliser_booleens(
            df_projection,
            [
                "historique_2024",
                "temp_ok",
                "precip_ok",
                "solar_ok",
                "soil_ok",
                "donnees_futures_completes",
                "apte",
            ],
        )

        df_green = df_projection.loc[
            df_projection["statut"] == "verte"
        ].copy()
        df_red = df_projection.loc[
            df_projection["statut"] == "rouge"
        ].copy()
        df_blue = df_projection.loc[
            df_projection["statut"] == "bleue"
        ].copy()

        nb_zones_2024 = int(df_projection["historique_2024"].sum())
        surface_reference = df_projection["surface_ha_12_5km"].sum()
        surface_maintenue = df_green["surface_future_estimee_ha"].sum()
        surface_perdue = df_red["surface_ha_12_5km"].sum()
        surface_nouvelle = df_blue["surface_nouvelle_estimee_ha"].sum()
        surface_future = df_projection["surface_future_estimee_ha"].sum()

        evolution_surface_ha = surface_future - surface_reference
        evolution_surface_pct = (
            evolution_surface_ha / surface_reference * 100
            if surface_reference > 0
            else np.nan
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric(
            "Zones historiques maintenues",
            f"{format_entier(surface_maintenue)} ha",
            help=(
                "Surface 2024 des mailles vertes, encore adaptées au scénario."
            ),
        )
        metric_1.caption(
            f"{format_entier(df_green['coordinates_id'].nunique())} zones"
        )

        metric_2.metric(
            f"Zones historiques perdues en {annee_sel_carte_2}",
            f"{format_entier(surface_perdue)} ha",
        )
        metric_2.caption(
            f"{format_entier(df_red['coordinates_id'].nunique())} zones"
        )

        metric_3.metric(
            f"Nouvelles zones aptes en {annee_sel_carte_2}",
            f"≈ {format_entier(surface_nouvelle)} ha",
            help=(
                "Surface potentielle estimée par l'ETL avec un coefficient "
                "de prudence de 0,5."
            ),
        )
        metric_3.caption(
            f"{format_entier(df_blue['coordinates_id'].nunique())} zones"
        )

        metric_4.metric(
            f"Évolution nette en {annee_sel_carte_2}",
            f"{evolution_surface_pct:+.1f} %"
            if pd.notna(evolution_surface_pct)
            else "Non disponible",
            delta=f"{format_entier(evolution_surface_ha)} ha",
        )

        st.caption(
            f"Référence 2024 : {format_entier(nb_zones_2024)} zones — "
            f"{format_entier(surface_reference)} ha."
        )

        def construire_hover(row: pd.Series, statut: str) -> str:
            titre = {
                "verte": "Zone historique maintenue",
                "rouge": "Zone historique perdue",
                "bleue": "Nouvelle zone apte",
            }[statut]

            surface_text = (
                f"<br>Surface 2024 : "
                f"{format_hover_value(row['surface_ha_12_5km'], 0, ' ha')}"
                if bool(row["historique_2024"])
                else (
                    f"<br>Surface potentielle estimée : "
                    f"{format_hover_value(row['surface_nouvelle_estimee_ha'], 0, ' ha')}"
                )
            )

            raison = row.get("raison_echec")
            raison_text = ""
            if statut == "rouge":
                raison_text = (
                    f"<br><b>Critères en échec : "
                    f"{raison if pd.notna(raison) and raison else 'non précisé'}</b>"
                )

            return (
                f"<b>{titre}</b>"
                f"{surface_text}"
                f"{raison_text}"
                f"<br>Sol : {row.get('soil_family_type', 'non disponible')} "
                f"{oui_non(row.get('soil_ok', False))}"
                f"<br>Température moyenne : "
                f"{format_hover_value(row.get('temp_moyenne_future'), 1, ' °C')} "
                f"{oui_non(row.get('temp_ok', False))}"
                f"<br>Mois le plus chaud : "
                f"{format_hover_value(row.get('temp_max_future'), 1, ' °C')}"
                f"<br>Précipitations : "
                f"{format_hover_value(row.get('precip_saison_future'), 0, ' mm')} "
                f"{oui_non(row.get('precip_ok', False))}"
                f"<br>Rayonnement : "
                f"{format_hover_value(row.get('rayonnement_saison_future'), 0, ' kWh/m²')} "
                f"{oui_non(row.get('solar_ok', False))}"
                f"<br>Harvest Score : "
                f"{format_hover_value(row.get('harvest_score_futur'), 1, '/100')}"
            )

        for frame, statut in [
            (df_green, "verte"),
            (df_red, "rouge"),
            (df_blue, "bleue"),
        ]:
            if not frame.empty:
                frame["hover_text"] = frame.apply(
                    construire_hover,
                    axis=1,
                    statut=statut,
                )

        surface_max = df_projection.loc[
            df_projection["historique_2024"],
            "surface_ha_12_5km",
        ].max()

        def tailles_points(frame: pd.DataFrame) -> np.ndarray:
            if frame.empty or pd.isna(surface_max) or surface_max <= 0:
                return np.array([])
            return 2 + np.sqrt(
                frame["surface_ha_12_5km"].clip(lower=0) / surface_max
            ) * 9

        fig_projection = go.Figure()

        # Ordre d'affichage : bleu, vert, rouge.
        if not df_blue.empty:
            fig_projection.add_trace(
                go.Scattermapbox(
                    lat=df_blue["lat"],
                    lon=df_blue["lon"],
                    mode="markers",
                    name=f"Nouvelle zone apte en {annee_sel_carte_2}",
                    marker={
                        "size": 8,
                        "color": COULEURS_STATUT["bleue"],
                        "opacity": 0.95,
                    },
                    text=df_blue["hover_text"],
                    customdata=df_blue[["coordinates_id"]].to_numpy(),
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        if not df_green.empty:
            fig_projection.add_trace(
                go.Scattermapbox(
                    lat=df_green["lat"],
                    lon=df_green["lon"],
                    mode="markers",
                    name="Zone historique maintenue",
                    marker={
                        "size": tailles_points(df_green),
                        "color": COULEURS_STATUT["verte"],
                        "opacity": 0.85,
                    },
                    text=df_green["hover_text"],
                    customdata=df_green[["coordinates_id"]].to_numpy(),
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        if not df_red.empty:
            fig_projection.add_trace(
                go.Scattermapbox(
                    lat=df_red["lat"],
                    lon=df_red["lon"],
                    mode="markers",
                    name=f"Zone historique perdue en {annee_sel_carte_2}",
                    marker={
                        "size": tailles_points(df_red),
                        "color": COULEURS_STATUT["rouge"],
                        "opacity": 0.40,
                    },
                    text=df_red["hover_text"],
                    customdata=df_red[["coordinates_id"]].to_numpy(),
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig_projection.update_layout(
            clickmode="event+select",
            hoverlabel={"font_size": 18},
            mapbox={
                "style": (
                    "https://basemaps.cartocdn.com/gl/"
                    "positron-nolabels-gl-style/style.json"
                ),
                "zoom": 4.3,
                "center": {"lat": 46.2, "lon": 2.2},
            },
            title=(
                f"Projection {annee_sel_carte_2} — "
                f"{culture_sel_carte_2}"
            ),
            height=650,
            margin={"r": 0, "t": 45, "l": 0, "b": 90},
            legend={
                "title": {"text": "Statut de projection"},
                "orientation": "h",
                "yanchor": "top",
                "y": -0.10,
                "xanchor": "center",
                "x": 0.5,
            },
        )

        map_event = st.plotly_chart(
            fig_projection,
            use_container_width=True,
            key="carte_projection_multicritere",
            on_select="rerun",
            selection_mode="points",
            config={"scrollZoom": False},
        )

        selected_coordinates_id = None
        if map_event and map_event.selection.points:
            customdata = map_event.selection.points[0].get("customdata")
            if customdata:
                selected_coordinates_id = int(customdata[0])

        if selected_coordinates_id is None:
            st.caption(
                "Cliquez sur une maille pour comparer son évolution de "
                "température à la moyenne nationale."
            )
        else:
            df_temp_nationale = load_national_temperature_evolution().rename(
                columns={"temperature": "Moyenne nationale"}
            )
            df_temp_locale = load_local_temperature_evolution(
                selected_coordinates_id
            ).rename(columns={"temperature": "Maille sélectionnée"})

            df_temperature = pd.merge(
                df_temp_nationale,
                df_temp_locale,
                on="year",
                how="outer",
            ).sort_values("year")

            fig_temperature = go.Figure()
            fig_temperature.add_trace(
                go.Scatter(
                    x=df_temperature["year"],
                    y=df_temperature["Moyenne nationale"],
                    mode="lines+markers",
                    name="Moyenne nationale",
                    line={"width": 4, "dash": "dashdot"},
                    marker={"size": 8},
                )
            )
            fig_temperature.add_trace(
                go.Scatter(
                    x=df_temperature["year"],
                    y=df_temperature["Maille sélectionnée"],
                    mode="lines+markers",
                    name=f"Maille {selected_coordinates_id}",
                    line={"width": 3},
                    marker={"size": 8},
                )
            )
            fig_temperature.update_layout(
                title="Évolution de la température moyenne d’avril à août",
                xaxis_title="Année",
                yaxis_title="Température moyenne (°C)",
                hovermode="x unified",
                height=420,
                margin={"l": 20, "r": 20, "t": 60, "b": 20},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
            fig_temperature.update_xaxes(
                tickmode="array",
                tickvals=[2024, 2054, 2084, 2100],
            )
            st.plotly_chart(fig_temperature, use_container_width=True)


# =========================================================
# MODULE DÉPARTEMENTAL
# =========================================================

st.markdown("---")

col_param_dep, col_carte_dep = st.columns([1, 3], gap="medium")

with col_param_dep:
    st.markdown("### 🎛️ Paramètres")

    culture_sel_dep = st.selectbox(
        "Sélectionnez la culture :",
        list(CULTURE_INFO.keys()),
        key="culture_sel_departements",
        format_func=lambda value: (
            f"{CULTURE_INFO[value]['picto']} {value}"
        ),
    )

    annee_sel_dep = st.radio(
        "Sélection du scénario",
        ANNEES_PROJECTION,
        horizontal=True,
        key="annee_sel_departements",
    )

    st.markdown("### 📊 Lecture de l'indice")
    st.markdown(
        """
        **Indice positif** : maintien important des surfaces et potentiel
        de nouvelles zones.

        **Indice proche de zéro** : les gains compensent partiellement
        les pertes.

        **Indice négatif** : vulnérabilité importante des surfaces
        historiques.
        """
    )

culture_id_dep = CULTURE_INFO[culture_sel_dep]["id"]

with col_carte_dep:
    st.subheader(
        f"🏛️ Indice départemental de vulnérabilité — {culture_sel_dep}"
    )

    try:
        df_departements = load_department_kpis(
            culture_id_dep,
            annee_sel_dep,
        ).copy()
        geojson_departements = load_departments_geojson()
    except Exception as error:
        st.error("Les indicateurs départementaux n'ont pas pu être chargés.")
        st.exception(error)
        df_departements = pd.DataFrame()

    if df_departements.empty:
        st.warning("Aucun indicateur départemental disponible.")
    else:
        df_departements["department_code"] = (
            df_departements["department_code"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(2)
        )

        fig_departements = px.choropleth(
            data_frame=df_departements,
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
                "surface_perdue_ha": ":,.0f",
                "surface_nouvelle_estimee_ha": ":,.0f",
                "surface_future_estimee_ha": ":,.0f",
                "evolution_surface_pct": ":.1f",
                "statut_departement": True,
            },
            labels={
                "indice": "Indice",
                "etoiles": "Évaluation",
                "surface_2024_ha": "Surface 2024 (ha)",
                "surface_maintenue_ha": "Surface maintenue (ha)",
                "surface_perdue_ha": "Surface perdue (ha)",
                "surface_nouvelle_estimee_ha": "Nouvelle surface (ha)",
                "surface_future_estimee_ha": "Surface future (ha)",
                "evolution_surface_pct": "Évolution (%)",
                "statut_departement": "Statut",
            },
            color_continuous_scale="RdYlGn",
            range_color=(-100, 100),
        )

        fig_departements.update_geos(fitbounds="locations", visible=False)
        fig_departements.update_layout(
            height=560,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            coloraxis_colorbar={
                "title": "Indice",
                "len": 0.72,
                "thickness": 18,
                "y": 0.5,
                "yanchor": "middle",
            },
        )
        fig_departements.update_traces(hoverlabel={"font_size": 18})

        st.plotly_chart(
            fig_departements,
            use_container_width=True,
            config={"scrollZoom": False, "displayModeBar": False},
        )


# =========================================================
# HARVEST SCORE — RÉSULTATS PRÉCALCULÉS
# =========================================================

st.markdown("---")

col_param_score, col_resultat_score = st.columns([1, 3], gap="medium")

with col_param_score:
    st.markdown("### 🎛️ Paramètres")

    culture_sel_score = st.selectbox(
        "Sélectionnez la culture :",
        list(CULTURE_INFO.keys()),
        key="culture_sel_harvest_score",
        format_func=lambda value: (
            f"{CULTURE_INFO[value]['picto']} {value}"
        ),
    )

    annee_sel_score = st.radio(
        "Sélection du scénario",
        ANNEES_PROJECTION,
        horizontal=True,
        key="annee_sel_harvest_score",
    )

    st.markdown("### 🧩 Composition du score")

    df_poids_score = pd.DataFrame(
        {
            "Paramètre": list(POIDS_HARVEST_SCORE.keys()),
            "Poids": list(POIDS_HARVEST_SCORE.values()),
        }
    )

    fig_poids_score = px.pie(
        df_poids_score,
        names="Paramètre",
        values="Poids",
        hole=0.45,
    )
    fig_poids_score.update_traces(
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>Poids dans le score : %{value} %"
            "<extra></extra>"
        ),
    )
    fig_poids_score.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 10, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.05,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    st.plotly_chart(
        fig_poids_score,
        use_container_width=True,
        config={"displayModeBar": False},
    )

culture_id_score = CULTURE_INFO[culture_sel_score]["id"]

with col_resultat_score:
    st.subheader(f"🌾 Harvest Score — {culture_sel_score}")
    st.caption(
        "Les valeurs sont lues directement dans Neon. Le rendement issu du "
        "modèle expérimental ne représente plus que 15 % du score."
    )

    try:
        df_kpi_national = load_national_kpis(
            culture_id_score,
            annee_sel_score,
        )
        df_kpi_departements = load_department_kpis(
            culture_id_score,
            annee_sel_score,
        )
    except Exception as error:
        st.error("Les KPIs n'ont pas pu être chargés depuis Neon.")
        st.exception(error)
        df_kpi_national = pd.DataFrame()
        df_kpi_departements = pd.DataFrame()

    if df_kpi_national.empty:
        st.warning("Aucun KPI disponible pour ce scénario.")
    else:
        kpi = df_kpi_national.iloc[0]

        score_national = kpi["harvest_score_national"]
        production_future = kpi["production_future_estimee_q"]
        evolution_production = kpi["evolution_production_pct"]
        surface_future = kpi["surface_future_estimee_ha"]
        evolution_surface = kpi["evolution_surface_pct"]
        rendement_futur = kpi["rendement_national_futur_q_ha"]
        evolution_rendement = kpi["evolution_rendement_pct"]

        score_1, score_2, score_3 = st.columns(3)

        score_1.metric(
            f"Harvest Score national {annee_sel_score}",
            f"{float(score_national):.1f} / 100"
            if pd.notna(score_national)
            else "Non disponible",
            help=(
                "Les zones rouges restent dans la surface de référence et "
                "contribuent avec zéro point."
            ),
        )

        score_2.metric(
            f"Production potentielle en {annee_sel_score}",
            f"{format_entier(production_future)} q",
            delta=(
                f"{float(evolution_production):+.1f} % vs 2024"
                if pd.notna(evolution_production)
                else None
            ),
        )

        score_3.metric(
            f"Surface future estimée en {annee_sel_score}",
            f"{format_entier(surface_future)} ha",
            delta=(
                f"{float(evolution_surface):+.1f} % vs 2024"
                if pd.notna(evolution_surface)
                else None
            ),
        )

        detail_1, detail_2, detail_3 = st.columns(3)
        detail_1.metric(
            "Rendement potentiel moyen",
            f"{format_decimal(rendement_futur, 1)} q/ha",
            delta=(
                f"{float(evolution_rendement):+.1f} % vs référence"
                if pd.notna(evolution_rendement)
                else None
            ),
        )
        detail_2.metric(
            "Mailles historiques maintenues",
            format_entier(kpi["mailles_vertes"]),
        )
        detail_3.metric(
            "Nouvelles mailles aptes",
            format_entier(kpi["mailles_bleues"]),
        )

        st.info(
            "Les rendements sont des estimations expérimentales. Les cartes "
            "d'aptitude verte, rouge et bleue reposent séparément sur les "
            "contraintes agronomiques et climatiques."
        )

        st.subheader("🏆 Classement départemental par rendement potentiel")

        classement = df_kpi_departements.loc[
            df_kpi_departements["surface_future_estimee_ha"] > 0
        ].copy()
        classement = classement.sort_values(
            "rendement_potentiel_q_ha",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
        classement["rang"] = classement.index + 1

        if classement.empty:
            st.warning("Aucun département ne possède une surface future positive.")
        else:
            df_affichage = classement[
                [
                    "rang",
                    "department_name",
                    "rendement_potentiel_q_ha",
                    "harvest_score",
                    "surface_future_estimee_ha",
                    "evolution_rendement_pct",
                ]
            ].rename(
                columns={
                    "rang": "Rang",
                    "department_name": "Département",
                    "rendement_potentiel_q_ha": "Rendement potentiel (q/ha)",
                    "harvest_score": "Harvest Score (/100)",
                    "surface_future_estimee_ha": "Surface future estimée (ha)",
                    "evolution_rendement_pct": "Évolution du rendement (%)",
                }
            )

            st.dataframe(
                df_affichage,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rang": st.column_config.NumberColumn(format="%d"),
                    "Rendement potentiel (q/ha)": (
                        st.column_config.NumberColumn(format="%.1f")
                    ),
                    "Harvest Score (/100)": (
                        st.column_config.NumberColumn(format="%.1f")
                    ),
                    "Surface future estimée (ha)": (
                        st.column_config.NumberColumn(format="%.0f")
                    ),
                    "Évolution du rendement (%)": (
                        st.column_config.NumberColumn(format="%+.1f %%")
                    ),
                },
            )

        if pd.notna(kpi["calculated_at"]):
            st.caption(
                f"Dernier calcul ETL enregistré dans Neon : "
                f"{kpi['calculated_at']}"
            )