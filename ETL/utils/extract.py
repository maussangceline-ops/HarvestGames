import os
from datetime import datetime
from pathlib import Path
import cdsapi
import boto3
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
if not BUCKET_NAME:
    raise ValueError(f"Variables d'environnement introuvables. Vérifie le chemin : {ENV_PATH}")

today_str = datetime.now().strftime("%Y%m%d")

client = cdsapi.Client(
    url=os.getenv("CDSAPI_URL"),
    key=os.getenv("CDSAPI_KEY")
)

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

AREA = [51.5, -5.5, 41, 9.8]  # France métropolitaine

# ==========================================
# 2. CONFIGURATION DES JEUX DE DONNÉES
# ==========================================
# Chaque entrée décrit un jeu de données Copernicus à extraire :
# - dataset       : nom du dataset CDS
# - variables     : liste des variables à télécharger (une requête par variable)
# - file_format   : format de sortie ("zip" ou "netcdf") -> détermine l'extension locale
# - file_prefix   : préfixe du nom de fichier local / clé S3
# - metadata      : métadonnées additionnelles attachées à l'objet S3
# - request_params: fonction qui construit le dict de requête pour une variable donnée

def build_cordex_request(variable):
    return {
        "domain": "europe",
        "experiment": "rcp_8_5",
        "horizontal_resolution": "0_11_degree_x_0_11_degree",
        "temporal_resolution": "monthly_mean",
        "variable": variable,
        "gcm_model": "cnrm_cerfacs_cm5",
        "rcm_model": "cnrm_aladin63",
        "ensemble_member": "r1i1p1",
        "year": ["2054", "2084", "2100"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "area": AREA,
        "format": "zip",
    }


def build_era5_request(variable):
    return {
        "product_type": "monthly_averaged_reanalysis",
        "variable": variable,
        "year": [str(y) for y in range(2010, 2025)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",  # Requis pour les moyennes mensuelles ERA5
        "area": AREA,
        "format": "netcdf",
    }


DATASETS = [
    {
        "name": "CORDEX",
        "dataset": "projections-cordex-domains-single-levels",
        "variables": [
            "2m_air_temperature",
            "mean_precipitation_flux",
            "surface_solar_radiation_downwards",
        ],
        "file_prefix": "raw_climate_cordex",
        "file_extension": "zip",
        "build_request": build_cordex_request,
        "metadata_extra": {"source": "copernicus_cds", "years": "2054_2084_2100"},
    },
    {
        "name": "ERA5",
        "dataset": "reanalysis-era5-single-levels-monthly-means",
        "variables": [
            "2m_temperature",
            "total_precipitation",
            "surface_solar_radiation_downwards",
        ],
        "file_prefix": "raw_era5_monthly",
        "file_extension": "nc",
        "build_request": build_era5_request,
        "metadata_extra": {"source": "copernicus_cds_era5_monthly", "years": "2010_2024"},
    },
]

# ==========================================
# 3. BOUCLE DE TÉLÉCHARGEMENT ET D'ENVOI S3
# ==========================================
for source in DATASETS:
    print(f"\n=== Traitement du jeu de données : {source['name']} ===")

    for variable in source["variables"]:
        print(f"\n--- Traitement de la variable : {variable} ---")

        local_file = f"{source['file_prefix']}_{variable}_{today_str}.{source['file_extension']}"
        s3_key = f"raw/{local_file}"
        request = source["build_request"](variable)

        try:
            # 1. Téléchargement depuis Copernicus
            print(f"1/3 Requête Copernicus en cours...")
            result = client.retrieve(source["dataset"], request)
            downloaded_path = Path(result.download(local_file))
            print(f"-> Fichier téléchargé : {downloaded_path.name}")

            # 2. Upload vers S3
            print(f"2/3 Envoi vers s3://{BUCKET_NAME}/{s3_key}...")
            s3.upload_file(
                str(downloaded_path),
                BUCKET_NAME,
                s3_key,
                ExtraArgs={
                    "Metadata": {
                        "dataset": source["dataset"],
                        "variable": variable,
                        **source["metadata_extra"],
                    }
                },
            )
            print(f"-> Succès de l'envoi S3.")

            # 3. Nettoyage local
            print("3/3 Nettoyage local...")
            if downloaded_path.exists():
                downloaded_path.unlink()

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {source['name']} / {variable} : {e}")
            continue  # Passe à la variable suivante même si l'une d'elles échoue

print("\n✅ Extraction CORDEX + ERA5 terminée !")