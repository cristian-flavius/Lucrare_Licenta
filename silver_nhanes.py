# Databricks notebook source
# ============================================================
# Corectii Silver Layer — adulti + Depression_Score valid
# ============================================================

from pyspark.sql.functions import col, when

df_silver = spark.table("catalog_licenta.default.bronze_nhanes")

# Selectam coloanele relevante
cols_needed = [
    "FeelingDownDepressedOrHopeless", "HaveLittleInterestInDoingThings",
    "FeelingBadAboutYourself", "FeelingTiredOrHavingLittleEnergy",
    "MovingOrSpeakingSlowlyOrTooFast", "TroubleConcentratingOnThings",
    "TroubleSleepingOrSleepingTooMuch", "PoorAppetiteOrOvereating",
    "HowOftenDoYouFeelDepressed", "HowOftenDoYouFeelWorriedOrAnxious",
    "TakeMedicationForDepression", "TakeMedicationForTheseFeelings",
    "EverToldDoctorHadTroubleSleeping", "HowOftenFeelOverlySleepyDuringDay",
    "AgeInYearsAtScreening", "Gender", "RacehispanicOrigin",
    "MaritalStatus", "EducationLevelAdults20", "AnnualHouseholdIncome",
    "MinutesSedentaryActivity", "VigorousRecreationalActivities",
    "ModerateRecreationalActivities", "WalkOrBicycle",
    "HaveSeriousDifficultyWalking", "SleepHoursWeekdaysOrWorkdays",
    "HowOftenDoYouSnore", "DietaryFiberGm_DR1TOT", "EnergyKcal_DR1TOT",
    "AlcoholGm_DR1TOT", "SmokedAtLeast100CigarettesInLife",
    "WeightKg", "WaistCircumferenceCm", "TrunkFatG",
    "Glycohemoglobin", "TotalCholesterolMgdl", "SystolicBloodPres1StRdgMmHg"
]

df_silver = df_silver.select(cols_needed)

# Eliminam minorii
df_silver = df_silver.filter(col("AgeInYearsAtScreening") >= 18)
print(f"După filtrare adulți (>=18): {df_silver.count()} rânduri")

# Tratam valorile aberante PHQ-9 (7/9 = refuz/nu știu → 0)
phq9_cols = [
    "FeelingDownDepressedOrHopeless", "HaveLittleInterestInDoingThings",
    "FeelingBadAboutYourself", "FeelingTiredOrHavingLittleEnergy",
    "MovingOrSpeakingSlowlyOrTooFast", "TroubleConcentratingOnThings",
    "TroubleSleepingOrSleepingTooMuch", "PoorAppetiteOrOvereating"
]

for c in phq9_cols:
    df_silver = df_silver.withColumn(c,
        when(col(c) > 3, None).otherwise(col(c)))

# Calculam Depression_Score
df_silver = df_silver.withColumn("Depression_Score",
    sum([col(c) for c in phq9_cols]))

# Eliminam randurile cu Depression_Score null
df_silver = df_silver.dropna(subset=["Depression_Score"])
print(f"După eliminare score null: {df_silver.count()} rânduri")

# Drop target nulls PHQ-2
df_silver = df_silver.dropna(subset=[
    "FeelingDownDepressedOrHopeless",
    "HaveLittleInterestInDoingThings"
])
print(f"După drop target nulls: {df_silver.count()} rânduri")

# Salvare
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("catalog_licenta.default.silver_nhanes_cleaned")

print(f"✅ Silver Layer corectat salvat!")
print(f"   Rânduri finale: {df_silver.count()}")
print(f"   Coloane: {len(df_silver.columns)}")
