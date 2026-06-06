# Databricks notebook source
# ==============================================================================
# BRONZE LAYER — Ingestia datelor brute NHANES 2017-2018
# ==============================================================================
# Sursa: CDC National Health and Nutrition Examination Survey (NHANES)
# Ciclu: 2017-2018 (sufixul _J in nomenclatura CDC)
# Acces: pachetul Python 'nhanes' descarca datele direct de pe serverele CDC
# ==============================================================================

# MAGIC %pip install nhanes

# COMMAND ----------

# 0. Pregatirea infrastructurii de stocare (catalog, schema, volum)
# Bronze este punctul de intrare al pipeline-ului, deci tot aici provizionam
# mediul Unity Catalog de care depind straturile urmatoare:
#   - catalog_licenta.default        — gazduieste tabelele Delta (bronze/silver/gold)
#   - volum_licenta                  — gazduieste modelul ML si figurile generate
#   - volum_licenta/figuri           — subdirectorul pentru vizualizari (PNG)
spark.sql("CREATE CATALOG IF NOT EXISTS catalog_licenta")
spark.sql("CREATE SCHEMA IF NOT EXISTS catalog_licenta.default")
spark.sql("CREATE VOLUME IF NOT EXISTS catalog_licenta.default.volum_licenta")

import os
os.makedirs("/Volumes/catalog_licenta/default/volum_licenta/figuri", exist_ok=True)

print("Infrastructura pregatita: catalog, schema, volum si subdirectorul figuri.")

# COMMAND ----------

from nhanes.load import load_NHANES_data

# 1. Descarcare date NHANES 2017-2018
print("Descarcare date NHANES 2017-2018...")
df_raw = load_NHANES_data(year='2017-2018')
print(f"Date descarcate: {df_raw.shape[0]} randuri, {df_raw.shape[1]} coloane.")

# 2. Conversie in Spark DataFrame
df_bronze = spark.createDataFrame(df_raw.reset_index())

# 3. Salvare ca Delta Table in catalog
df_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("catalog_licenta.default.bronze_nhanes")

print(f"Bronze Layer complet!")
print(f"   Randuri: {df_bronze.count()}")
print(f"   Coloane: {len(df_bronze.columns)}")
