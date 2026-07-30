import os
from pathlib import Path
import boto3
import dotenv
import pandas as pd
import xarray as xr

ENV_PATH = Path(__file__).resolve().parent / ".env"
dotenv.load_dotenv(dotenv_path=ENV_PATH)

BUCKET_NAME = os.getenv("AWS_S3_BUCKET")

# Nom exact du fichier déjà présent dans bronze/
NC_FILENAME = "tas_EUR-11_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_mon_20540116-20541216.nc"
BRONZE_KEY = f"bronze/{NC_FILENAME}"

LOCAL_NC = Path(__file__).resolve().parent / NC_FILENAME

BRONZE_CSV = "cordex_2054_raw.csv"
SILVER_PARQUET = "cordex_2054_clean.parquet"
SILVER_CSV = "cordex_2054_clean.csv"

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

# 1. Récupération du .nc depuis BRONZE (S3) vers local
print("1/5 Téléchargement du .nc depuis S3 bronze/...")
s3.download_file(BUCKET_NAME, BRONZE_KEY, str(LOCAL_NC))
print(f"-> Récupéré localement : {LOCAL_NC.name}")

# 2. Conversion NetCDF -> DataFrame brut
print("2/5 Conversion NetCDF -> DataFrame brut...")
ds = xr.open_dataset(LOCAL_NC)
df_raw = ds.to_dataframe().reset_index()
ds.close()

# 3. Upload du CSV BRUT (optionnel, pour audit/debug — déjà en bronze via le .nc)
print("3/5 Upload du CSV brut dans BRONZE (optionnel)...")
df_raw.to_csv(BRONZE_CSV, index=False)
s3.upload_file(BRONZE_CSV, BUCKET_NAME, f"bronze/{BRONZE_CSV}")
print(f"-> Déposé : s3://{BUCKET_NAME}/bronze/{BRONZE_CSV}")

# 4. Transformations pour la zone SILVER
print("4/5 Nettoyage et transformation pour SILVER...")

possible_temp_cols = ["t2m", "tas", "2m_air_temperature", "tasmax"]
temp_col = [col for col in df_raw.columns if col in possible_temp_cols][0]

df_silver = df_raw.copy()
df_silver["temp_celsius"] = (df_silver[temp_col] - 273.15).round(2)
df_silver = df_silver.dropna(subset=["temp_celsius"])

df_silver = df_silver.drop_duplicates(subset=["time", "lat", "lon"])

# Note : lat/lon ici sont des variables 2D (grille rotated pole), pas des dims simples
cols_to_keep = [col for col in ["time", "lat", "lon", "latitude", "longitude", "temp_celsius"] if col in df_silver.columns]
df_silver = df_silver[cols_to_keep]

# 5. Upload des fichiers transformés dans SILVER
print("5/5 Upload des fichiers optimisés dans SILVER...")
df_silver.to_parquet(SILVER_PARQUET, index=False)
s3.upload_file(SILVER_PARQUET, BUCKET_NAME, f"silver/{SILVER_PARQUET}")

df_silver.to_csv(SILVER_CSV, index=False)
s3.upload_file(SILVER_CSV, BUCKET_NAME, f"silver/{SILVER_CSV}")

print(f"-> Déposé : s3://{BUCKET_NAME}/silver/{SILVER_PARQUET}")

# Nettoyage local
for file in [LOCAL_NC, BRONZE_CSV, SILVER_PARQUET, SILVER_CSV]:
    if os.path.exists(file):
        os.remove(file)

print("-> Transformation bronze -> silver terminée avec succès !")