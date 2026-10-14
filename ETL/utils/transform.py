# transform.py
import os
import zipfile
import boto3
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from dotenv import load_dotenv
from scipy.spatial import cKDTree

# 1. Configuration globale
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
BUCKET = os.getenv("AWS_S3_BUCKET")
s3_client = boto3.client('s3')

TMP_DIR = "tmp_data"
os.makedirs(TMP_DIR, exist_ok=True)

def s3(path):
    return f"s3://{BUCKET}/{path}"

def download_from_s3(s3_key, local_filename):
    """Télécharge un fichier de S3 vers le dossier tmp_data local."""
    print(f"   Téléchargement de {s3_key}...")
    local_path = os.path.join(TMP_DIR, local_filename)
    s3_client.download_file(BUCKET, s3_key, local_path)
    return local_path

# =====================================================================
# PIPELINE 1 : CLIMAT PASSÉ (ERA5)
# =====================================================================
def process_climate_past():
    print("\n=== DÉBUT CLIMAT PASSÉ (ERA5) ===")
    
    # --- ÉTAPE RAW -> BRONZE ---
    print(">> ERA5 : Raw vers Bronze...")
    
    nc_files = {
        "raw/raw_era5_monthly_2m_temperature_20260729.nc": "era5_temp.nc",
        "raw/raw_era5_monthly_total_precipitation_20260729.nc": "era5_precip.nc",
        "raw/raw_era5_monthly_surface_solar_radiation_downwards_20260729.nc": "era5_rad.nc"
    }
    
    datasets = []
    vars_to_drop = ['time_bnds', 'number', 'expver']
    
    for s3_key, local_name in nc_files.items():
        local_path = download_from_s3(s3_key, local_name)
        with xr.open_dataset(local_path) as ds:
            drop_list = [v for v in vars_to_drop if v in ds.variables]
            datasets.append(ds.drop_vars(drop_list).load())
        os.remove(local_path) 
            
    merged_ds = xr.merge(datasets)
    df = merged_ds.to_dataframe().reset_index()
    df.to_csv(s3("bronze/bronze_climate_era5.csv"), index=False)
    print(">> Bronze sauvegardé !")

    # --- ÉTAPE BRONZE -> SILVER ---
    print(">> ERA5 : Bronze vers Silver...")
    df_bronze = pd.read_csv(s3("bronze/bronze_climate_era5.csv"))
    
    time_col = 'valid_time' if 'valid_time' in df_bronze.columns else 'time'
    df_bronze = df_bronze.rename(columns={
        time_col: 'time',
        't2m': 'temp_celsius',
        'tp': 'precip_mm',
        'ssrd': 'solar_rad'
    })

    df_bronze['time'] = pd.to_datetime(df_bronze['time'])
    df_bronze['year'] = df_bronze['time'].dt.year
    df_bronze['month'] = df_bronze['time'].dt.month

    if df_bronze['temp_celsius'].mean() > 200:
        df_bronze['temp_celsius'] = df_bronze['temp_celsius'] - 273.15

    days_in_month = df_bronze['time'].dt.days_in_month
    df_bronze['precip_mm'] = df_bronze['precip_mm'] * 1000 * days_in_month
    df_bronze['solar_rad'] = df_bronze['solar_rad'] / 86400

    LAT_MIN, LAT_MAX = 41.0, 52.0
    LON_MIN, LON_MAX = -5.0, 10.0
    mask = (
        (df_bronze['latitude'] >= LAT_MIN) & (df_bronze['latitude'] <= LAT_MAX) &
        (df_bronze['longitude'] >= LON_MIN) & (df_bronze['longitude'] <= LON_MAX)
    )
    df_filtered = df_bronze[mask].copy()

    df_silver = df_filtered.groupby(
        ['latitude', 'longitude', 'year', 'month'], as_index=False
    ).max()

    colonnes_a_garder = [
        'year', 'month', 'latitude', 'longitude',
        'temp_celsius', 'precip_mm', 'solar_rad'
    ]
    df_silver = df_silver[colonnes_a_garder].copy()

    df_silver['year'] = df_silver['year'].astype('int16')
    df_silver['month'] = df_silver['month'].astype('int8')
    df_silver['temp_celsius'] = df_silver['temp_celsius'].astype('float32')
    df_silver['precip_mm'] = df_silver['precip_mm'].astype('float32')
    df_silver['solar_rad'] = df_silver['solar_rad'].astype('float32')

    df_silver.to_parquet(s3("silver/silver_climate_era5.parquet"), index=False)
    print(">> Silver sauvegardé !")

    # --- ÉTAPE SILVER -> GOLD ---
    print(">> ERA5 : Silver vers Gold...")
    df_era5 = pd.read_parquet(s3("silver/silver_climate_era5.parquet"))
    
    # Lecture de dim_coordinates depuis le dossier raw/ sur S3
    dim_coords = pd.read_csv(s3("raw/dim_coordinates.csv"), sep=';')
    dim_coords = dim_coords.rename(columns={'lat': 'latitude', 'lon': 'longitude'})

    if 'coordinates_id' not in dim_coords.columns:
        raise KeyError("La table dim_coordinates doit contenir la colonne 'coordinates_id'.")

    dim_coords_unique = dim_coords.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
    print(f"   Points uniques dans dim_coordinates : {len(dim_coords_unique)}")

    era5_points = df_era5[['latitude', 'longitude']].drop_duplicates()
    cordex_points = dim_coords_unique[['latitude', 'longitude']].drop_duplicates()

    era5_coords = era5_points[['latitude', 'longitude']].to_numpy()
    cordex_coords = cordex_points[['latitude', 'longitude']].to_numpy()

    print("   Recherche du point ERA5 le plus proche pour chaque point CORDEX (cKDTree)...")
    tree = cKDTree(era5_coords)
    distances, indices = tree.query(cordex_coords, k=1)

    mapping_data = []
    for i, (lat_c, lon_c) in enumerate(cordex_coords):
        idx = indices[i]
        lat_e, lon_e = era5_coords[idx]
        mapping_data.append((lat_c, lon_c, lat_e, lon_e))

    mapping_df = pd.DataFrame(mapping_data, columns=['lat_cordex', 'lon_cordex', 'lat_era5', 'lon_era5'])

    print("   Jointure pour associer ERA5 aux points CORDEX...")
    df_mapped = pd.merge(
        mapping_df, df_era5,
        left_on=['lat_era5', 'lon_era5'],
        right_on=['latitude', 'longitude'],
        how='left'
    )
    df_gold = pd.merge(
        df_mapped,
        dim_coords_unique[['latitude', 'longitude', 'coordinates_id']],
        left_on=['lat_cordex', 'lon_cordex'],
        right_on=['latitude', 'longitude'],
        how='left',
        suffixes=('', '_cordex')
    )

    nb_missing = df_gold['coordinates_id'].isna().sum()
    if nb_missing > 0:
        print(f" ⚠️ {nb_missing} lignes sans coordinates_id.")
    else:
        print("   Tous les points ont un coordinates_id.")

    df_gold = df_gold.drop(columns=['latitude_cordex', 'longitude_cordex'], errors='ignore')

    print("   Renommage et mise en forme Gold...")
    df_gold = df_gold.rename(columns={
        'temp_celsius': 'temp_mean_past',
        'precip_mm': 'precip_past',
        'solar_rad': 'solar_rad_past'
    })
    df_gold['date'] = pd.to_datetime(dict(year=df_gold['year'], month=df_gold['month'], day=1))

    colonnes_finales = [
        'coordinates_id', 'date', 'year', 'month',
        'temp_mean_past', 'precip_past', 'solar_rad_past'
    ]
    df_gold = df_gold[colonnes_finales]
    df_gold['coordinates_id'] = df_gold['coordinates_id'].astype('int32')
    df_gold['year'] = df_gold['year'].astype('int16')
    df_gold['month'] = df_gold['month'].astype('int8')

    print(">> Sauvegarde Gold sur S3...")
    df_gold.to_parquet(s3("gold/src_climate_past.parquet"), index=False)
    print(">> Gold sauvegardé !")

# =====================================================================
# PIPELINE 2 : CLIMAT FUTUR (CORDEX)
# =====================================================================
def process_climate_future():
    print("\n=== DÉBUT CLIMAT FUTUR (CORDEX) ===")
    
    # --- ÉTAPE RAW -> BRONZE ---
    print(">> CORDEX : Raw vers Bronze...")
    # ⚠️ VÉRIFIE QUE CES NOMS CORRESPONDENT EXACTEMENT À TON BUCKET S3
    cordex_zips = [
        "raw/raw_climate_cordex_2m_air_temperature_20260727.zip",
        "raw/raw_climate_cordex_mean_precipitation_flux_20260727.zip",
        "raw/raw_climate_cordex_surface_solar_radiation_downwards_20260727.zip"
    ]
    datasets = []
    vars_to_drop = ['Lambert_Conformal', 'time_bounds', 'bounds_lat', 'bounds_lon', 'axis_nbounds', 'nvertex', 'string1', 'height']
    
    for s3_key in cordex_zips:
        local_zip = download_from_s3(s3_key, os.path.basename(s3_key))
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.nc'):
                    zip_ref.extract(file_info, TMP_DIR)
                    local_nc_path = os.path.join(TMP_DIR, file_info.filename)
                    with xr.open_dataset(local_nc_path) as ds:
                        drop_list = [v for v in vars_to_drop if v in ds.variables]
                        datasets.append(ds.drop_vars(drop_list).load())
                    os.remove(local_nc_path)
        os.remove(local_zip)
        
    merged_ds = xr.merge(datasets)
    df = merged_ds.to_dataframe().reset_index()
    df.to_csv(s3("bronze/bronze_climate.csv"), index=False)
    print(">> Bronze CORDEX sauvegardé !")

    # --- GÉNÉRATION DE DIM_COORDINATES ---
    print(">> Génération de dim_coordinates à partir du maillage CORDEX...")
    df_bronze = pd.read_csv(s3("bronze/bronze_climate.csv"))
    
    # On filtre la boîte englobante de la France AVANT de prendre les points uniques
    LAT_MIN, LAT_MAX = 41.0, 51.5
    LON_MIN, LON_MAX = -5.5, 9.8
    mask_france = (
        (df_bronze['lat'] >= LAT_MIN) & (df_bronze['lat'] <= LAT_MAX) &
        (df_bronze['lon'] >= LON_MIN) & (df_bronze['lon'] <= LON_MAX)
    )
    df_bronze_france = df_bronze[mask_france].copy()
    
    dim_coordinates = (
        df_bronze_france[["lat", "lon"]] # On utilise le dataframe filtré
        .drop_duplicates()
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )
    dim_coordinates["coordinates_id"] = dim_coordinates.index + 1
    dim_coordinates.to_csv(s3("raw/dim_coordinates.csv"), index=False, sep=";")
    print(f">> dim_coordinates.csv sauvegardé ({len(dim_coordinates)} points) !")

    # --- ÉTAPE BRONZE -> SILVER ---
    print(">> CORDEX : Bronze vers Silver...")
    df_coords = dim_coordinates.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    df_coords = df_coords[['latitude', 'longitude']]
    
    # 1. Renommage
    df_bronze = df_bronze.rename(columns={
        'lat': 'latitude',
        'lon': 'longitude',
        'tas': 'temp_celsius',
        'pr': 'precip_mm',
        'rsds': 'solar_rad'
    })

    # 2. Gestion des dates
    df_bronze['time'] = pd.to_datetime(df_bronze['time'])
    df_bronze['year'] = df_bronze['time'].dt.year
    df_bronze['month'] = df_bronze['time'].dt.month

    # 3. Conversions des unités
    if df_bronze['temp_celsius'].mean() > 200:
        df_bronze['temp_celsius'] = df_bronze['temp_celsius'] - 273.15

    days_in_month = df_bronze['time'].dt.days_in_month
    df_bronze['precip_mm'] = df_bronze['precip_mm'] * 86400 * days_in_month

    # 4. Filtrage spatial
    df_silver = pd.merge(df_bronze, df_coords, on=['latitude', 'longitude'], how='inner')
    print(f"-> Lignes après filtrage : {len(df_silver):,}".replace(',', ' '))

    colonnes_a_garder = ['year', 'month', 'latitude', 'longitude', 'temp_celsius', 'precip_mm', 'solar_rad']
    df_silver = df_silver[colonnes_a_garder]

    df_silver['year'] = df_silver['year'].astype('int16')
    df_silver['month'] = df_silver['month'].astype('int8')
    df_silver['temp_celsius'] = df_silver['temp_celsius'].astype('float32')
    df_silver['precip_mm'] = df_silver['precip_mm'].astype('float32')
    df_silver['solar_rad'] = df_silver['solar_rad'].astype('float32')
    
    df_silver.to_parquet(s3("silver/silver_climate_cordex.parquet"), index=False)
    print(">> Silver CORDEX sauvegardé !")

    # --- ÉTAPE SILVER -> GOLD ---
    print(">> CORDEX : Silver vers Gold...")
    df_silver = pd.read_parquet(s3("silver/silver_climate_cordex.parquet"))
    
    dim_coords = dim_coordinates.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    dim_coords = dim_coords[['latitude', 'longitude', 'coordinates_id']]

    df_gold = pd.merge(df_silver, dim_coords, on=['latitude', 'longitude'], how='left')

    nb_sans_id = df_gold['coordinates_id'].isna().sum()
    if nb_sans_id > 0:
        print(f" ⚠️ ATTENTION : {nb_sans_id} lignes n'ont pas trouvé de coordinates_id !")

    df_gold = df_gold.rename(columns={
        'temp_celsius': 'temp_mean_future',
        'precip_mm': 'precip_future',
        'solar_rad': 'solar_rad_future'
    })

    df_gold['date'] = pd.to_datetime(dict(year=df_gold['year'], month=df_gold['month'], day=1))
    df_gold = df_gold.drop(columns=['latitude', 'longitude'])

    colonnes_finales = ['coordinates_id', 'date', 'year', 'month', 'temp_mean_future', 'precip_future', 'solar_rad_future']
    df_gold = df_gold[colonnes_finales]
    df_gold['coordinates_id'] = df_gold['coordinates_id'].astype('int32')

    df_gold.to_parquet(s3("gold/src_climate_future.parquet"), index=False)
    print(">> Gold CORDEX sauvegardé !")

# =====================================================================
# PIPELINE 3 : RENDEMENTS AGRICOLES (SAA)
# =====================================================================
def process_yields():
    print("\n=== DÉBUT RENDEMENTS (SAA) ===")
    
    # --- ÉTAPE RAW -> BRONZE ---
    print(">> Yields : Raw vers Bronze...")
    local_xlsx = download_from_s3("raw/SAA_2010-2025_provisoires_donnees_departementales.xlsx", "SAA.xlsx")
    sheets = {"COP": "bronze_saa_cop.csv", "PDT": "bronze_saa_pdt.csv", "VIG": "bronze_saa_vig.csv"}
    
    for sheet_name, bronze_file in sheets.items():
        df = pd.read_excel(local_xlsx, sheet_name=sheet_name, header=5)
        df.to_csv(s3(f"bronze/{bronze_file}"), index=False, sep=";")
    
    os.remove(local_xlsx) # Nettoyage

    # --- ÉTAPE BRONZE -> SILVER ---
    print(">> Yields : Bronze vers Silver...")
    
    SOURCES = [
        {"file": s3("bronze/bronze_saa_vig.csv"), "libelle": "02 - Vigne à raisin de cuve", "culture_id": 1, "sep": ";"},
        {"file": s3("bronze/bronze_saa_cop.csv"), "libelle": "43 - Pois chiches (y compris semences)", "culture_id": 2, "sep": ";"},
        {"file": s3("bronze/bronze_saa_pdt.csv"), "libelle": "06 - Pommes de terre de consommation (04 + 05)", "culture_id": 3, "sep": ";"},
    ]

    ANNEES = list(range(2010, 2026))

    def parse_lib_dep(lib_dep: str):
        code, _, nom = lib_dep.partition(" - ")
        return code.strip(), nom.strip()

    def wide_to_long(df: pd.DataFrame, culture_id: int) -> pd.DataFrame:
        df = df.copy()
        dup = df.duplicated(subset=["LIB_DEP"], keep=False)
        if dup.any():
            print(f"   ATTENTION : {dup.sum()} lignes dupliquées sur LIB_DEP pour culture_id={culture_id}")

        df[["code_departement", "nom_departement"]] = df["LIB_DEP"].apply(
            lambda x: pd.Series(parse_lib_dep(x))
        )

        records = []
        for annee in ANNEES:
            col_rend = f"REND_{annee}"
            col_surf = f"SURF_{annee}" if f"SURF_{annee}" in df.columns else f"SURF_PROD_{annee}"
            col_prod = f"PROD_{annee}"

            if col_rend not in df.columns:
                continue

            sub = df[["code_departement", "nom_departement", "STATUT_QUALITE", col_rend, col_surf, col_prod]].copy()
            
            # On renomme directement avec les bons noms pour l'étape Gold
            sub = sub.rename(columns={
                col_rend: "rendement_q_ha", 
                col_surf: "surface_ha", 
                col_prod: "production_q",
                "STATUT_QUALITE": "statut_qualite"
            })
            sub["annee"] = annee
            records.append(sub)

        long_df = pd.concat(records, ignore_index=True)
        long_df["culture_id"] = culture_id
        return long_df

    print("Chargement et filtrage des 3 sources depuis S3...")
    frames = []
    for source in SOURCES:
        print(f" - {source['file']} | libelle = '{source['libelle']}'")
        df = pd.read_csv(source["file"], sep=source["sep"], dtype={"LIB_DEP": str})

        df_filtre = df[df["LIB_SAA"] == source["libelle"]].copy()
        print(f"   {len(df_filtre)} lignes après filtrage")

        if len(df_filtre) == 0:
            continue

        frames.append(wide_to_long(df_filtre, source["culture_id"]))

    silver_yields = pd.concat(frames, ignore_index=True)
    silver_yields = silver_yields[
        ["code_departement", "nom_departement", "culture_id", "annee", "rendement_q_ha", "surface_ha", "production_q", "statut_qualite"]
    ]

    silver_yields.to_csv(s3("silver/src_yields_2010_2025.csv"), index=False, sep=";")
    print(">> Silver sauvegardé !")

    # --- ÉTAPE SILVER -> GOLD ---
    print(">> Yields : Silver vers Gold...")
    
    # 1. Chargement des inputs depuis S3
    src_yields = pd.read_csv(s3("silver/src_yields_2010_2025.csv"), sep=";", dtype={"code_departement": str})
    dim_coords = pd.read_csv(s3("raw/dim_coordinates.csv"), sep=';')
    src_cultures = pd.read_csv(s3("silver/src_cultures_2024.csv"), sep=";")

    def normaliser_code_dept(code) -> str:
        code = str(code).strip()
        if code and code[-1] in ("A", "B") and len(code) <= 3:
            return code[:-1].zfill(2) + code[-1]
        return code.zfill(3)

    print("1. Préparation de la grille CORDEX (points uniques)...")
    df_grid = dim_coords.copy()
    df_grid["lat"] = df_grid["lat"].round(6)
    df_grid["lon"] = df_grid["lon"].round(6)
    df_grid = df_grid[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    print(f"   {len(df_grid):,} points")

    gdf_grid = gpd.GeoDataFrame(
        df_grid,
        geometry=gpd.points_from_xy(df_grid["lon"], df_grid["lat"]),
        crs="EPSG:4326",
    )

    print("2. Récupération du découpage administratif des départements depuis S3...")
    local_geojson = download_from_s3("raw/departements.geojson", "departements.geojson")
    gdf_dept = gpd.read_file(local_geojson)[["code", "nom", "geometry"]]
    os.remove(local_geojson) # Nettoyage

    print("3. Attribution du département à chaque point CORDEX...")
    grid_dept = gpd.sjoin(gdf_grid, gdf_dept, how="left", predicate="intersects")
    grid_dept = grid_dept.rename(columns={"code": "code_dept_grid", "nom": "nom_dept_grid"})
    grid_dept["code_dept_grid"] = grid_dept["code_dept_grid"].apply(
        lambda c: normaliser_code_dept(c) if pd.notna(c) else c
    )

    nb_sans_dept = grid_dept["code_dept_grid"].isna().sum()
    print(f"   {nb_sans_dept:,} / {len(grid_dept):,} points sans département (mer, etc.)")

    grid_dept = grid_dept[["lat", "lon", "code_dept_grid", "nom_dept_grid"]].drop_duplicates(subset=["lat", "lon"])

    print("4. Récupération de coordinates_id...")
    dim_coords["lat"] = dim_coords["lat"].round(6)
    dim_coords["lon"] = dim_coords["lon"].round(6)
    grid_dept = pd.merge(grid_dept, dim_coords, on=["lat", "lon"], how="left")
    grid_dept = grid_dept.drop(columns=["lat", "lon"])

    print("5. Préparation des rendements (toutes années 2010-2025)...")
    src_yields["code_departement"] = src_yields["code_departement"].apply(normaliser_code_dept)

    print("6. Fusion : grille -> présence culture -> rendement département...")
    gold = pd.merge(src_cultures, grid_dept, on="coordinates_id", how="left")

    gold = pd.merge(
        gold,
        src_yields[["code_departement", "culture_id", "annee", "rendement_q_ha", "surface_ha", "production_q", "statut_qualite"]],
        left_on=["code_dept_grid", "culture_id"],
        right_on=["code_departement", "culture_id"],
        how="left",
        suffixes=("_grille", "_dept"),
    )
    print(f"   {len(gold):,} lignes après jointure")

    print("7. Filtrage : rendement affiché uniquement là où la culture est réellement présente...")
    if 'nb_parcelles_12_5km' in gold.columns:
        masque_absence = gold["nb_parcelles_12_5km"] == 0
        cols_a_annuler = ["rendement_q_ha", "surface_ha", "production_q", "statut_qualite"]
        gold.loc[masque_absence, cols_a_annuler] = None

    print("8. Nettoyage final...")
    gold = gold.drop(columns=["code_departement", "nom_dept_grid"], errors="ignore")
    gold = gold.rename(columns={"code_dept_grid": "code_departement"})

    # Sauvegarde finale sur S3
    gold.to_csv(s3("gold/gold_src_yields_cordex_2010_2025.csv"), index=False, sep=";")
    print(">> Gold sauvegardé !")
    
# =====================================================================
# PIPELINE 4 : CULTURES (RAW -> SILVER/GOLD)
# =====================================================================
def process_cultures():
    print("\n=== DÉBUT CULTURES (GPKG) ===")
    
    # --- ÉTAPE RAW -> GOLD ---
    print(">> Cultures : Raw vers Silver/Gold...")
    local_gpkg = download_from_s3("raw/cultures_selectionnees_2024.gpkg", "cultures.gpkg")
    
    # 1. Chargement de la grille de référence depuis S3
    dim_coords = pd.read_csv(s3("raw/dim_coordinates.csv"), sep=';', float_precision="round_trip")
    dim_coords["lat"] = dim_coords["lat"].round(6)
    dim_coords["lon"] = dim_coords["lon"].round(6)
    
    gdf_cordex = gpd.GeoDataFrame(
        dim_coords,
        geometry=gpd.points_from_xy(dim_coords["lon"], dim_coords["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=2154)

    print("2. Chargement du fichier vectoriel lourd des parcelles (.gpkg)...")
    gdf_cultures = gpd.read_file(local_gpkg).to_crs(epsg=2154)

    CULTURE_MAPPING = {
        "Vigne (raisins de cuve)": 1,
        "Pois chiche": 2,
        "Pomme de terre de consommation": 3,
    }
    
    # On gère les majuscules/minuscules selon le fichier
    col_culture = "NOM_CULTURE" if "NOM_CULTURE" in gdf_cultures.columns else "LIB_CULTURE"
    gdf_cultures["culture_id"] = gdf_cultures[col_culture].map(CULTURE_MAPPING)
    gdf_cultures = gdf_cultures.dropna(subset=["culture_id"])
    gdf_cultures["culture_id"] = gdf_cultures["culture_id"].astype(int)

    # Calcul de la surface exacte
    gdf_cultures["surface_ha_exacte"] = gdf_cultures.geometry.area / 10000

    print("3. Jointure spatiale : attribution du point CORDEX le plus proche (sjoin_nearest)...")
    gold_parcelles = gpd.sjoin_nearest(
        gdf_cultures,
        gdf_cordex[["coordinates_id", "geometry"]],
        how="left",
        distance_col="dist_point_cordex_m",
    )
    gold_parcelles = gold_parcelles[~gold_parcelles.index.duplicated(keep="first")]

    print("4. Nettoyage et exportation Gold...")
    cols_to_keep = [
        "ID_PARCEL", "culture_id", col_culture, "surface_ha_exacte", 
        "coordinates_id", "geometry"
    ]
    gold_parcelles = gold_parcelles[[c for c in cols_to_keep if c in gold_parcelles.columns]]
    
    # Sauvegarde du Parquet Gold (pour Neon DB)
    gold_parcelles.to_parquet(s3("gold/gold_parcelles_cultures_2024.parquet"))
    print(">> gold_parcelles_cultures_2024.parquet sauvegardé !")

    print("5. Génération de src_cultures_2024.csv (agrégation par coordinates_id)...")
    # Pour que le script des rendements fonctionne, on a besoin de compter les parcelles
    # autour de chaque point (ici on compte simplement les parcelles associées au point)
    src_cultures = (
        gold_parcelles.groupby(["coordinates_id", "culture_id"])
        .size()
        .reset_index(name="nb_parcelles_12_5km")
    )
    # On s'assure que tous les points x cultures existent (même si 0 parcelle)
    all_points = gdf_cordex[["coordinates_id"]].drop_duplicates()
    all_cultures = pd.DataFrame({"culture_id": [1, 2, 3]})
    base = all_points.merge(all_cultures, how="cross")
    
    src_cultures = pd.merge(base, src_cultures, on=["coordinates_id", "culture_id"], how="left").fillna(0)
    src_cultures["nb_parcelles_12_5km"] = src_cultures["nb_parcelles_12_5km"].astype(int)
    
    # Sauvegarde du CSV Silver
    src_cultures.to_csv(s3("silver/src_cultures_2024.csv"), index=False, sep=";")
    print(">> silver/src_cultures_2024.csv sauvegardé !")
    
    os.remove(local_gpkg) # Nettoyage du fichier lourd local

# =====================================================================
# PIPELINE 5 : SOLS (RAW -> GOLD)
# =====================================================================
def process_soils():
    print("\n=== DÉBUT SOLS ===")
    
    # --- ÉTAPE GOLD -> GOLD ---
    print(">> Sols : Raw vers Gold...")
    
    # 1. Chargement des inputs depuis S3
    print("1. Chargement des inputs...")
    df_yields = pd.read_csv(s3("gold/gold_src_yields_cordex_2010_2025.csv"), sep=";", float_precision="round_trip", dtype={"code_departement": str}, low_memory=False)
    dim_coordinates = pd.read_csv(s3("raw/dim_coordinates.csv"), sep=";", float_precision="round_trip")
    
    # Lecture directe du CSV des sols sur S3
    cols_sols = ["latitude", "longitude", "famille_agro", "libelle_texture"]
    df_sols = pd.read_csv(s3("raw/sols_france_agricoles.csv"), sep=",", usecols=cols_sols)
    df_sols = df_sols.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    def normaliser_code_dept(code) -> str:
        code = str(code).strip()
        if not code:
            return code
        if code[-1].upper() in ("A", "B") and len(code) <= 3:
            return code[:-1].zfill(2) + code[-1].upper()
        return code.zfill(3)

    # 2. Préparation de la grille CORDEX
    print("2. Préparation de la grille CORDEX...")
    df_grid = dim_coordinates[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    df_grid["lat"] = df_grid["lat"].round(6)
    df_grid["lon"] = df_grid["lon"].round(6)

    # 3. Matching spatial par cKDTree
    print("3. Matching spatial (cKDTree)...")
    lat_moyenne_france = 46.5
    facteur_correction = np.cos(np.radians(lat_moyenne_france))

    coords_sols_corrigees = df_sols[["latitude", "longitude"]].copy()
    coords_sols_corrigees["longitude"] = coords_sols_corrigees["longitude"] * facteur_correction
    tree = cKDTree(coords_sols_corrigees.values)

    coords_cordex_corrigees = df_grid[["lat", "lon"]].copy()
    coords_cordex_corrigees["lon"] = coords_cordex_corrigees["lon"] * facteur_correction

    distances, indices = tree.query(coords_cordex_corrigees.values, k=1)

    df_grid["soil_family_type"] = df_sols.loc[indices, "famille_agro"].values
    df_grid["soil_type"] = df_sols.loc[indices, "libelle_texture"].values

    seuil_km = 10
    seuil_degres_lat = seuil_km / 111
    mask_trop_loin = distances > seuil_degres_lat
    df_grid.loc[mask_trop_loin, ["soil_family_type", "soil_type"]] = None
    print(f"   Points sans sol pertinent (> {seuil_km} km) : {mask_trop_loin.sum()} / {len(df_grid)}")

    # 4. Récupération du référentiel département depuis S3
    print("4. Chargement du référentiel département...")
    local_geojson = download_from_s3("raw/departements.geojson", "departements.geojson")
    gdf_dept = gpd.read_file(local_geojson)[["code", "nom"]].copy()
    os.remove(local_geojson) # Nettoyage
    
    gdf_dept["code"] = gdf_dept["code"].apply(normaliser_code_dept)
    gdf_dept = gdf_dept.rename(columns={"code": "department_code", "nom": "department_name"})

    # 5. Récupération du department_code depuis la table yields (gold)
    print("5. Récupération département depuis src_yields...")
    dim_coordinates["lat"] = dim_coordinates["lat"].round(6)
    dim_coordinates["lon"] = dim_coordinates["lon"].round(6)

    df_yields_latlon = pd.merge(df_yields, dim_coordinates, on="coordinates_id", how="left")
    df_yields_latlon = df_yields_latlon.rename(columns={"code_departement": "department_code"})
    df_yields_latlon["department_code"] = df_yields_latlon["department_code"].apply(normaliser_code_dept)

    dept_info = df_yields_latlon[["lat", "lon", "department_code"]].drop_duplicates()
    dept_info["lat"] = dept_info["lat"].round(6)
    dept_info["lon"] = dept_info["lon"].round(6)

    # 6. Assemblage de dim_soils
    print("6. Assemblage de dim_soils...")
    dim_soils = pd.merge(df_grid, dept_info, on=["lat", "lon"], how="left")
    dim_soils = pd.merge(dim_soils, gdf_dept, on="department_code", how="left")
    dim_soils = dim_soils.drop_duplicates(subset=["lat", "lon"])

    # 7. Ajout de la région
    print("7. Ajout de la région...")
    DEPT_TO_REGION = {
        "01": ("84", "Auvergne-Rhône-Alpes"), "02": ("32", "Hauts-de-France"),
        "03": ("84", "Auvergne-Rhône-Alpes"), "04": ("93", "Provence-Alpes-Côte d'Azur"),
        "05": ("93", "Provence-Alpes-Côte d'Azur"), "06": ("93", "Provence-Alpes-Côte d'Azur"),
        "07": ("84", "Auvergne-Rhône-Alpes"), "08": ("44", "Grand Est"),
        "09": ("76", "Occitanie"), "10": ("44", "Grand Est"),
        "11": ("76", "Occitanie"), "12": ("76", "Occitanie"),
        "13": ("93", "Provence-Alpes-Côte d'Azur"), "14": ("28", "Normandie"),
        "15": ("84", "Auvergne-Rhône-Alpes"), "16": ("75", "Nouvelle-Aquitaine"),
        "17": ("75", "Nouvelle-Aquitaine"), "18": ("24", "Centre-Val de Loire"),
        "19": ("75", "Nouvelle-Aquitaine"), "21": ("27", "Bourgogne-Franche-Comté"),
        "22": ("53", "Bretagne"), "23": ("75", "Nouvelle-Aquitaine"),
        "24": ("75", "Nouvelle-Aquitaine"), "25": ("27", "Bourgogne-Franche-Comté"),
        "26": ("84", "Auvergne-Rhône-Alpes"), "27": ("28", "Normandie"),
        "28": ("24", "Centre-Val de Loire"), "29": ("53", "Bretagne"),
        "2A": ("94", "Corse"), "2B": ("94", "Corse"),
        "30": ("76", "Occitanie"), "31": ("76", "Occitanie"),
        "32": ("76", "Occitanie"), "33": ("75", "Nouvelle-Aquitaine"),
        "34": ("76", "Occitanie"), "35": ("53", "Bretagne"),
        "36": ("24", "Centre-Val de Loire"), "37": ("24", "Centre-Val de Loire"),
        "38": ("84", "Auvergne-Rhône-Alpes"), "39": ("27", "Bourgogne-Franche-Comté"),
        "40": ("75", "Nouvelle-Aquitaine"), "41": ("24", "Centre-Val de Loire"),
        "42": ("84", "Auvergne-Rhône-Alpes"), "43": ("84", "Auvergne-Rhône-Alpes"),
        "44": ("52", "Pays de la Loire"), "45": ("24", "Centre-Val de Loire"),
        "46": ("76", "Occitanie"), "47": ("75", "Nouvelle-Aquitaine"),
        "48": ("76", "Occitanie"), "49": ("52", "Pays de la Loire"),
        "50": ("28", "Normandie"), "51": ("44", "Grand Est"),
        "52": ("44", "Grand Est"), "53": ("52", "Pays de la Loire"),
        "54": ("44", "Grand Est"), "55": ("44", "Grand Est"),
        "56": ("53", "Bretagne"), "57": ("44", "Grand Est"),
        "58": ("27", "Bourgogne-Franche-Comté"), "59": ("32", "Hauts-de-France"),
        "60": ("32", "Hauts-de-France"), "61": ("28", "Normandie"),
        "62": ("32", "Hauts-de-France"), "63": ("84", "Auvergne-Rhône-Alpes"),
        "64": ("75", "Nouvelle-Aquitaine"), "65": ("76", "Occitanie"),
        "66": ("76", "Occitanie"), "67": ("44", "Grand Est"),
        "68": ("44", "Grand Est"), "69": ("84", "Auvergne-Rhône-Alpes"),
        "70": ("27", "Bourgogne-Franche-Comté"), "71": ("27", "Bourgogne-Franche-Comté"),
        "72": ("52", "Pays de la Loire"), "73": ("84", "Auvergne-Rhône-Alpes"),
        "74": ("84", "Auvergne-Rhône-Alpes"), "75": ("11", "Île-de-France"),
        "76": ("28", "Normandie"), "77": ("11", "Île-de-France"),
        "78": ("11", "Île-de-France"), "79": ("75", "Nouvelle-Aquitaine"),
        "80": ("32", "Hauts-de-France"), "81": ("76", "Occitanie"),
        "82": ("76", "Occitanie"), "83": ("93", "Provence-Alpes-Côte d'Azur"),
        "84": ("93", "Provence-Alpes-Côte d'Azur"), "85": ("52", "Pays de la Loire"),
        "86": ("75", "Nouvelle-Aquitaine"), "87": ("75", "Nouvelle-Aquitaine"),
        "88": ("44", "Grand Est"), "89": ("27", "Bourgogne-Franche-Comté"),
        "90": ("27", "Bourgogne-Franche-Comté"), "91": ("11", "Île-de-France"),
        "92": ("11", "Île-de-France"), "93": ("11", "Île-de-France"),
        "94": ("11", "Île-de-France"), "95": ("11", "Île-de-France"),
    }

    DEPT_TO_REGION_NORM = {normaliser_code_dept(k): v for k, v in DEPT_TO_REGION.items()}

    dim_soils["region_code"] = dim_soils["department_code"].map(
        lambda c: DEPT_TO_REGION_NORM.get(str(c).strip(), (None, None))[0]
    )
    dim_soils["region_name"] = dim_soils["department_code"].map(
        lambda c: DEPT_TO_REGION_NORM.get(str(c).strip(), (None, None))[1]
    )

    # 8. Jointure finale avec coordinates_id
    print("8. Jointure finale avec coordinates_id...")
    dim_soils = pd.merge(dim_soils, dim_coordinates[["lat", "lon", "coordinates_id"]],
                        on=["lat", "lon"], how="left")

    dim_soils = dim_soils.drop(columns=["lat", "lon"])
    dim_soils["region_code"] = dim_soils["region_code"].astype(str).replace("None", None)

    # Sauvegarde finale sur S3
    dim_soils.to_csv(s3("gold/dim_soils.csv"), index=False, sep=";")
    print(">> Gold sauvegardé !")

# =====================================================================
# LE CHEF D'ORCHESTRE
# =====================================================================
def transform_all():
    print("====================================")
    print("DÉBUT DE TOUTES LES TRANSFORMATIONS")
    print("====================================")
    
    # 1. Climat (Génère dim_coordinates.csv)
    process_climate_future()
    process_climate_past()
    
    # 2. Cultures (Génère silver/src_cultures_2024.csv)
    process_cultures()
    
    # 3. Rendements (Utilise src_cultures_2024.csv)
    process_yields()
    
    # 4. Sols (Utilise gold_src_yields_cordex_2010_2025.csv)
    process_soils()
    
    print("====================================")
    print("TOUTES LES TRANSFORMATIONS TERMINÉES")
    print("====================================")

if __name__ == "__main__":
    transform_all()