# Databricks notebook source
# ============================================================
# GOLD LAYER — Agregări statistice bazate pe PHQ-9 corectat
# ============================================================

from pyspark.sql.functions import col, when, avg, count, round as _round

df_silver = spark.table("catalog_licenta.default.silver_nhanes_cleaned")

# 1. Clasificare severitate depresie conform standardului PHQ-9
df_gold = df_silver.withColumn("Depression_Severity",
    when(col("Depression_Score") <= 4,  "Minimala")
    .when(col("Depression_Score") <= 9,  "Usoara")
    .when(col("Depression_Score") <= 14, "Moderata")
    .when(col("Depression_Score") <= 19, "Moderat_Severa")
    .otherwise("Severa")
)

# 2. Distributia severității
print("=== Distributie severitate depresie ===")
display(df_gold.groupBy("Depression_Severity")
    .agg(count("*").alias("Total_Pacienti"))
    .orderBy("Total_Pacienti", ascending=False))

# 3. Biomarkeri clinici per categorie de severitate
print("=== Biomarkeri clinici per severitate ===")
display(df_gold.groupBy("Depression_Severity")
    .agg(
        _round(avg("Glycohemoglobin"), 2).alias("Avg_HbA1c"),
        _round(avg("TotalCholesterolMgdl"), 1).alias("Avg_Cholesterol"),
        _round(avg("SystolicBloodPres1StRdgMmHg"), 1).alias("Avg_Systolic"),
        _round(avg("WeightKg"), 1).alias("Avg_Weight"),
        _round(avg("WaistCircumferenceCm"), 1).alias("Avg_Waist")
    )
    .orderBy("Avg_HbA1c", ascending=False))

# 4. Factori comportamentali per severitate
print("=== Factori comportamentali per severitate ===")
display(df_gold.groupBy("Depression_Severity")
    .agg(
        _round(avg("SleepHoursWeekdaysOrWorkdays"), 2).alias("Avg_Sleep"),
        _round(avg("MinutesSedentaryActivity"), 1).alias("Avg_Sedentary_Min"),
        _round(avg("AlcoholGm_DR1TOT"), 2).alias("Avg_Alcohol"),
        _round(avg("AgeInYearsAtScreening"), 1).alias("Avg_Age")
    )
    .orderBy("Avg_Sedentary_Min", ascending=False))

# 5. Salvare Gold Delta Table
df_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("catalog_licenta.default.gold_nhanes")

print(f"✅ Gold Layer complet!")
print(f"   Rânduri: {df_gold.count()}")
