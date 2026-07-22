import os
import zipfile
from pathlib import Path
import cdsapi
import boto3
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LOCAL_FILE = 'cordex_2054_temp_download' # extension ajoutée dynamiquement
EXTRACT_DIR = Path('cordex_extract') # dossier pour stockage local
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")

if not BUCKET_NAME:
    raise ValueError(f"Variables d'environnement introuvables. Vérifie le chemin : {ENV_PATH}")

client = cdsapi.Client(
    url=os.getenv("CDSAPI_URL"),
    key=os.getenv("CDSAPI_KEY")
)

dataset = "projections-cordex-domains-single-levels"
request = {
    "domain": "europe",
    "experiment": "rcp_8_5",
    "horizontal_resolution": "0_11_degree_x_0_11_degree",
    "temporal_resolution": "monthly_mean",
    "variable": "2m_air_temperature",
    "gcm_model": "cnrm_cerfacs_cm5",
    "rcm_model": "cnrm_aladin63",
    "ensemble_member": "r1i1p1",
    "year": ["2054"],
    "month": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ],
    "area": [51.5, -5.5, 41, 9.8]
}

print("1/4 Lancement du téléchargement Copernicus...")
result = client.retrieve(dataset, request)
downloaded_path = result.download(LOCAL_FILE)
downloaded_path = Path(downloaded_path)
print(f"-> Fichier téléchargé localement : {downloaded_path.name}")

print("2/4 Vérification du format et extraction si besoin...")
EXTRACT_DIR.mkdir(exist_ok=True)

if downloaded_path.suffix == '.zip' or zipfile.is_zipfile(downloaded_path):
    with zipfile.ZipFile(downloaded_path, 'r') as z:
        nc_files = [f for f in z.namelist() if f.endswith('.nc')]
        if not nc_files:
            raise RuntimeError("Aucun fichier .nc trouvé dans l'archive téléchargée.")
        z.extractall(EXTRACT_DIR)
    nc_path = EXTRACT_DIR / nc_files[0]
elif downloaded_path.suffix == '.nc':
    nc_path = downloaded_path
else:
    raise RuntimeError(f"Format de fichier inattendu : {downloaded_path.suffix}")

nc_filename = nc_path.name
print(f"-> Fichier .nc prêt : {nc_filename}")

print("3/4 Envoi vers le bucket AWS S3...")
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

s3_key = f"bronze/{nc_filename}"
s3.upload_file(str(nc_path), BUCKET_NAME, s3_key)
print(f"-> Succès : Fichier uploadé dans s3://{BUCKET_NAME}/{s3_key}")

print("4/4 Nettoyage local...")
if downloaded_path.exists() and downloaded_path != nc_path:
    downloaded_path.unlink()
if nc_path.exists() and nc_path.is_relative_to(EXTRACT_DIR):
    nc_path.unlink()
    if EXTRACT_DIR.exists() and not any(EXTRACT_DIR.iterdir()):
        EXTRACT_DIR.rmdir()
print("-> Terminé !")