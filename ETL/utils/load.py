# load.py
import io
import os
import boto3
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
NEON_DB_URL = os.getenv("NEON_DB_URL")

# --- Configuration unifiée de tous les fichiers ---
# ⚠ On ajoute le suffixe _v2 pour ne pas écraser les tables actuelles de ta collègue
files_to_load = {
    "dim_constraints_v2": {"key": "raw/dim_constraints.csv", "sep": ";"},
    "dim_coordinates_v2": {"key": "raw/dim_coordinates.csv", "sep": ";"}, # Généré dans raw/ par transform.py
    "dim_soils_v2": {"key": "gold/dim_soils.csv", "sep": ";"},
    "src_cultures_plots_2024_v2": {
        "key": "gold/gold_parcelles_cultures_2024.parquet",
        "format": "geoparquet",
    },
    "src_climate_future_v2": {"key": "gold/src_climate_future.parquet", "format": "parquet"},
    "src_climate_past_v2": {"key": "gold/src_climate_past.parquet", "format": "parquet"},
    "src_yields_cordex_2010_2025_v2": {
        "key": "gold/gold_src_yields_cordex_2010_2025.csv", # Nouveau nom exact
        "sep": ";",
    },
}

# On met à jour les clés du dictionnaire avec le suffixe _v2
DTYPE_OVERRIDES = {
    "dim_soils_v2": {"department_code": str, "region_code": str},
    "src_yields_cordex_2010_2025_v2": {"code_departement": str}, # transform.py utilise code_departement
}

print("**Connexion aux services...**")
s3_client = boto3.client("s3")
engine = create_engine(NEON_DB_URL)

# Activer PostGIS (nécessaire pour le geoparquet)
# ⚠ On NE SUPPRIME PAS les vues existantes pour ne pas casser le Streamlit actuel
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
    conn.commit()
print("Extension PostGIS activée.")

# --- Chargement depuis S3 et injection dans Neon ---
for table_name, file_info in files_to_load.items():
    print(f"\nTéléchargement de {file_info['key']} depuis S3...")
    obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_info["key"])
    buffer = io.BytesIO(obj["Body"].read())

    fmt = file_info.get("format")

    # 1. GeoParquet (données spatiales avec géométrie)
    if fmt == "geoparquet":
        gdf = gpd.read_parquet(buffer)

        if "coordinates_id" in gdf.columns:
            nb_null = gdf["coordinates_id"].isna().sum()
            if nb_null > 0:
                print(f"  ATTENTION : {nb_null} lignes avec coordinates_id manquant dans {table_name}")

        print(f"Injection spatiale (to_postgis) dans Neon (Table: {table_name}) - {len(gdf):,} lignes...")
        gdf.to_postgis(
            name=table_name,
            con=engine,
            if_exists="replace", # Crée la nouvelle table _v2
            index=False,
            chunksize=5000,
        )

    # 2. Parquet classique (non spatial)
    elif fmt == "parquet":
        df = pd.read_parquet(buffer)

        if "coordinates_id" in df.columns:
            nb_null = df["coordinates_id"].isna().sum()
            if nb_null > 0:
                print(f"  ATTENTION : {nb_null} lignes avec coordinates_id manquant dans {table_name}")

        print(f"Injection dans Neon DB (Table: {table_name}) - {len(df):,} lignes...")
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

    # 3. CSV
    else:
        df = pd.read_csv(
            buffer,
            sep=file_info.get("sep", ";"),
            float_precision="round_trip",
            dtype=DTYPE_OVERRIDES.get(table_name, {}),
            low_memory=False, # Pour éviter le DtypeWarning
        )

        if "coordinates_id" in df.columns:
            nb_null = df["coordinates_id"].isna().sum()
            if nb_null > 0:
                print(f"  ATTENTION : {nb_null} lignes avec coordinates_id manquant dans {table_name}")

        print(f"Injection dans Neon DB (Table: {table_name}) - {len(df):,} lignes...")
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

print("\n**Succès ! Données gold injectées dans Neon DB (Tables _v2).**")