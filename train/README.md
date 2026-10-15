# Train — Modèle ML & table de faits

Ce module contient `build_fact_crop_projection.py`, qui :

1. charge les données climatiques et de rendement depuis Neon ;
2. entraîne un **RandomForestRegressor** par culture
   (variables : température, précipitations, rayonnement, type de sol) ;
3. calcule les statuts agronomiques et le **Harvest Score** par maille ;
4. écrit la table `fact_crop_projection` lue par Streamlit.

C'est le pont entre l'ETL (`../ETL/`) et la dataviz (`../streamlit/`).

## ⚠️ Avertissement

Ce script **écrase** les partitions `fact_crop_projection`
(culture × année) dans Neon. **Ne pas exécuter** sans stratégie
d'isolation (branche Neon dédiée, projet séparé, Postgres local) :
la base actuelle est celle que Streamlit Cloud lit en production.

## Utilisation sûre

```bash
# Calcule tout mais n'écrit RIEN dans Neon
python build_fact_crop_projection.py --dry-run

# Traite une seule partition (culture 1, année 2054)
python build_fact_crop_projection.py --culture-id 1 --year 2054