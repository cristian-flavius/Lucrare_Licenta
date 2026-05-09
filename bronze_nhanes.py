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
