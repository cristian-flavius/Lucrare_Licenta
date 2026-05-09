# Databricks notebook source
import subprocess
subprocess.run(["pip", "install", "nhanes"], capture_output=True)

from nhanes.load import load_NHANES_data

# 1. Descarcam datele NHANES 2017-2018
print("Descărcare date NHANES 2017-2018...")
df_raw = load_NHANES_data(year='2017-2018')
print(f"Date descărcate: {df_raw.shape[0]} rânduri, {df_raw.shape[1]} coloane.")

# 2. Conversie în Spark DataFrame
df_bronze = spark.createDataFrame(df_raw.reset_index())

# 3. Salvare ca Delta Table în catalog
df_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("catalog_licenta.default.bronze_nhanes")

print(f"✅ Bronze Layer complet!")
print(f"   Rânduri: {df_bronze.count()}")
print(f"   Coloane: {len(df_bronze.columns)}")
