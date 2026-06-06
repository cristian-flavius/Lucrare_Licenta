# Databricks notebook source
# ==============================================================================
# SILVER LAYER — Selectie, curatare si calcul scor depresie PHQ-9
# ==============================================================================
# Input:  catalog_licenta.default.bronze_nhanes (date brute NHANES 2017-2018)
# Output: catalog_licenta.default.silver_nhanes (date curatate, scor PHQ-9)
#
# Transformari aplicate:
#   1. Selectia a 37 coloane relevante din cele ~197 disponibile
#   2. Filtrarea adultilor (>=18 ani) — PHQ-9 este validat clinic doar pentru adulti
#   3. Tratarea valorilor aberante PHQ-9 (codurile 7/9 din NHANES = refuz/nu stiu)
#   4. Calculul scorului total Depression_Score (suma celor 8 itemi PHQ-9, rang 0-24)
#   5. Eliminarea randurilor fara scor valid
# ==============================================================================

from pyspark.sql.functions import col, when

# COMMAND ----------

df = spark.table("catalog_licenta.default.bronze_nhanes")

# 1. Selectam 37 coloane relevante grupate pe categorii
cols_needed = [
    # Sanatate mintala — cei 8 itemi PHQ-9 (0=deloc, 1=cateva zile, 2=>jumatate, 3=aproape zilnic)
    "FeelingDownDepressedOrHopeless", "HaveLittleInterestInDoingThings",
    "FeelingBadAboutYourself", "FeelingTiredOrHavingLittleEnergy",
    "MovingOrSpeakingSlowlyOrTooFast", "TroubleConcentratingOnThings",
    "TroubleSleepingOrSleepingTooMuch", "PoorAppetiteOrOvereating",
    # Sanatate mintala — alte variabile (nu intra in scorul PHQ-9)
    "HowOftenDoYouFeelDepressed", "HowOftenDoYouFeelWorriedOrAnxious",
    "TakeMedicationForDepression", "TakeMedicationForTheseFeelings",
    "EverToldDoctorHadTroubleSleeping", "HowOftenFeelOverlySleepyDuringDay",
    # Demografice
    "AgeInYearsAtScreening", "Gender", "RacehispanicOrigin",
    "MaritalStatus", "EducationLevelAdults20", "AnnualHouseholdIncome",
    # Activitate fizica
    "MinutesSedentaryActivity", "VigorousRecreationalActivities",
    "ModerateRecreationalActivities", "WalkOrBicycle",
    "HaveSeriousDifficultyWalking",
    # Somn
    "SleepHoursWeekdaysOrWorkdays", "HowOftenDoYouSnore",
    # Dieta
    "DietaryFiberGm_DR1TOT", "EnergyKcal_DR1TOT", "AlcoholGm_DR1TOT",
    # Fumat
    "SmokedAtLeast100CigarettesInLife",
    # Antropometrice
    "WeightKg", "WaistCircumferenceCm", "TrunkFatG",
    # Clinice (biomarkeri)
    "Glycohemoglobin", "TotalCholesterolMgdl", "SystolicBloodPres1StRdgMmHg"
]

df = df.select(cols_needed)
print(f"Coloane selectate: {len(df.columns)}")

# COMMAND ----------

# 2. Filtrare adulti — PHQ-9 este validat clinic doar pentru persoane >= 18 ani
df = df.filter(col("AgeInYearsAtScreening") >= 18)
print(f"Dupa filtrare adulti (>=18): {df.count()} randuri")

# COMMAND ----------

# 3. Tratare valori aberante PHQ-9
# In NHANES, codurile 7 si 9 inseamna "refuz" si "nu stiu" — NU sunt raspunsuri valide.
# Le inlocuim cu None (null) pentru a nu introduce erori in scor.
phq9_cols = [
    "FeelingDownDepressedOrHopeless", "HaveLittleInterestInDoingThings",
    "FeelingBadAboutYourself", "FeelingTiredOrHavingLittleEnergy",
    "MovingOrSpeakingSlowlyOrTooFast", "TroubleConcentratingOnThings",
    "TroubleSleepingOrSleepingTooMuch", "PoorAppetiteOrOvereating"
]

for c in phq9_cols:
    df = df.withColumn(c, when(col(c) > 3, None).otherwise(col(c)))

# COMMAND ----------

# 3b. Tratare valori invalide pentru variabilele continue din chestionar
# PAD680 (MinutesSedentaryActivity) foloseste codurile 7777="refuz" si
# 9999="nu stiu". Cum o zi are maximum 1440 de minute, orice valoare peste
# acest prag este imposibila fiziologic si reprezinta un cod-santinela
# necurat. Le inlocuim cu None; vor fi imputate ulterior cu mediana in ml_nhanes.
df = df.withColumn("MinutesSedentaryActivity",
    when(col("MinutesSedentaryActivity") > 1440, None)
    .otherwise(col("MinutesSedentaryActivity")))

# Validare: confirmam ca nu mai exista valori imposibile dupa curatare
from pyspark.sql.functions import max as _max, min as _min, count as _count
stats = df.select(
    _min("MinutesSedentaryActivity").alias("min"),
    _max("MinutesSedentaryActivity").alias("max"),
    _count(when(col("MinutesSedentaryActivity").isNull(), 1)).alias("nuluri")
).collect()[0]
print(f"Sedentarism dupa curatare — min: {stats['min']}, max: {stats['max']}, valori nule: {stats['nuluri']}")

# COMMAND ----------

# 4. Calculul scorului total PHQ-9 (Depression_Score)
# Suma celor 8 itemi: rang posibil 0-24 (8 itemi x max 3 puncte)
# Daca oricare item este null, suma devine null — randurile respective se elimina
df = df.withColumn("Depression_Score", sum([col(c) for c in phq9_cols]))

# 5. Eliminare randuri fara scor valid
df = df.dropna(subset=["Depression_Score"])
print(f"Randuri cu scor PHQ-9 valid: {df.count()}")

# COMMAND ----------

# 6. Salvare ca Delta Table
df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("catalog_licenta.default.silver_nhanes")

print(f"Silver Layer complet!")
print(f"   Randuri finale: {df.count()}")
print(f"   Coloane: {len(df.columns)}")
