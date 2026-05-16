# Databricks notebook source
# ==============================================================================
# GOLD LAYER — Clasificare severitate si agregari statistice
# ==============================================================================
# Input:  catalog_licenta.default.silver_nhanes (date curatate cu Depression_Score)
# Output: catalog_licenta.default.gold_nhanes  (date cu categorie de severitate)
#
# Transformari aplicate:
#   1. Clasificare severitate depresie pe 5 nivele conform pragurilor clinice PHQ-9
#   2. Agregari statistice: biomarkeri si factori comportamentali per categorie
#   3. Salvare tabel Gold pregatit pentru modelul ML
# ==============================================================================

from pyspark.sql.functions import col, when, avg, count, round as _round

# COMMAND ----------

df = spark.table("catalog_licenta.default.silver_nhanes")

# 1. Clasificare severitate depresie conform standardului clinic PHQ-9
# Praguri: 0-4 Minimala, 5-9 Usoara, 10-14 Moderata, 15-19 Moderat-Severa, 20-24 Severa
df = df.withColumn("Depression_Severity",
    when(col("Depression_Score") <= 4,  "Minimala")
    .when(col("Depression_Score") <= 9,  "Usoara")
    .when(col("Depression_Score") <= 14, "Moderata")
    .when(col("Depression_Score") <= 19, "Moderat_Severa")
    .otherwise("Severa")
)

# COMMAND ----------

# 2. Distributia severitatii
print("=== Distributie severitate depresie ===")
display(df.groupBy("Depression_Severity")
    .agg(count("*").alias("Total_Pacienti"))
    .orderBy("Total_Pacienti", ascending=False))

# COMMAND ----------

# 3. Biomarkeri clinici per categorie de severitate
print("=== Biomarkeri clinici per severitate ===")
display(df.groupBy("Depression_Severity")
    .agg(
        _round(avg("Glycohemoglobin"), 2).alias("Avg_HbA1c"),
        _round(avg("TotalCholesterolMgdl"), 1).alias("Avg_Cholesterol"),
        _round(avg("SystolicBloodPres1StRdgMmHg"), 1).alias("Avg_Systolic"),
        _round(avg("WeightKg"), 1).alias("Avg_Weight"),
        _round(avg("WaistCircumferenceCm"), 1).alias("Avg_Waist")
    )
    .orderBy("Avg_HbA1c", ascending=False))

# COMMAND ----------

# 4. Factori comportamentali per severitate
print("=== Factori comportamentali per severitate ===")
display(df.groupBy("Depression_Severity")
    .agg(
        _round(avg("SleepHoursWeekdaysOrWorkdays"), 2).alias("Avg_Sleep"),
        _round(avg("MinutesSedentaryActivity"), 1).alias("Avg_Sedentary_Min"),
        _round(avg("AlcoholGm_DR1TOT"), 2).alias("Avg_Alcohol"),
        _round(avg("AgeInYearsAtScreening"), 1).alias("Avg_Age")
    )
    .orderBy("Avg_Sedentary_Min", ascending=False))

# COMMAND ----------

# 5. Salvare Gold Delta Table
df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("catalog_licenta.default.gold_nhanes")

print(f"Gold Layer complet!")
print(f"   Randuri: {df.count()}")