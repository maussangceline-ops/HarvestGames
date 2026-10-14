# ETL — Extract / Transform / Load

Pipeline ETL : S3 (Copernicus) → transformations → Neon.

## Scripts

- `utils/extract.py` — extraction depuis S3
- `utils/transform.py` — nettoyage bronze → silver
- `utils/load.py` — écriture vers Neon

## Statut

Scripts en place. Pipeline complet non re-testé de A à Z (Phase 2).

⚠️ **Ne pas exécuter `load.py`** contre la base Neon de production —
la stratégie d'isolation de test est prévue en Phase 2.