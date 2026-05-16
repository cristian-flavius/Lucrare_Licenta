# Databricks notebook source
# ==============================================================================
# ML LAYER — Antrenare model Random Forest pentru clasificarea severitatii depresiei
# ==============================================================================
# Input:  catalog_licenta.default.gold_nhanes (date cu Depression_Severity)
# Output: Model RandomForest salvat pe volum
#
# Nota tehnica: Folosim scikit-learn in locul Spark MLlib deoarece Databricks
# Free Edition foloseste Spark Connect, care restrictioneaza anumite operatii
# MLlib. Cu 5070 de randuri, sklearn este oricum alegerea optima ca performanta
# si simplitate. Pipeline-ul Medallion (Bronze/Silver/Gold) ramane pe
# Spark + Delta Lake.
#
# Etape:
#   1. Conversia tabelei Gold in pandas DataFrame
#   2. Encoding variabile categorice (string -> numeric)
#   3. Selectia feature-urilor cu excluderea coloanelor cu data leakage
#   4. Imputare valori lipsa cu mediana
#   5. Antrenare Random Forest Classifier (multi-class: 5 nivele)
#   6. Evaluare pe set de test (Accuracy, F1, Precision, Recall)
#   7. Analiza importantei feature-urilor
#   8. Salvare model si encoderi pentru reutilizare
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# COMMAND ----------

# 1. Incarcare date din Gold (Spark) si conversie in pandas
df_spark = spark.table("catalog_licenta.default.gold_nhanes")
df = df_spark.toPandas()
print(f"Date incarcate: {len(df)} randuri, {len(df.columns)} coloane")

# COMMAND ----------

# 2. Encoding variabile categorice (string -> numeric)
# LabelEncoder atribuie cate un index numeric fiecarei valori distincte.
# Salvam encoderii pentru a putea reproduce encoding-ul in alte notebook-uri.
string_cols = [
    "Gender", "RacehispanicOrigin", "MaritalStatus",
    "EducationLevelAdults20", "HowOftenDoYouSnore",
    "HowOftenFeelOverlySleepyDuringDay",
    "EverToldDoctorHadTroubleSleeping",
    "SmokedAtLeast100CigarettesInLife",
    "WalkOrBicycle", "HaveSeriousDifficultyWalking",
    "VigorousRecreationalActivities",
    "ModerateRecreationalActivities",
    "AnnualHouseholdIncome"
]

encoders = {}
for c in string_cols:
    # Inlocuim valorile null cu "UNKNOWN" pentru a putea aplica encoder-ul
    df[c] = df[c].fillna("UNKNOWN").astype(str)
    le = LabelEncoder()
    df[f"{c}_Idx"] = le.fit_transform(df[c])
    encoders[c] = le
    df = df.drop(columns=[c])

# Encoding label-ului (target): Depression_Severity -> numeric
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Depression_Severity"])
print(f"Clase target: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# COMMAND ----------

# 3. Selectia feature-urilor
# EXCLUDEM urmatoarele categorii pentru a evita data leakage:
#   - Depression_Severity, Depression_Score, label = derivate direct din target
#   - Cei 8 itemi PHQ-9 individuali = componentele scorului (ar fi trivial sa
#     prezici severitatea din propriile sale componente)
#   - TakeMedicationForDepression, TakeMedicationForTheseFeelings = consecinte
#     ale diagnosticului, nu cauze
#   - HowOftenDoYouFeelDepressed, HowOftenDoYouFeelWorriedOrAnxious = masuratori
#     directe ale starii care defineste target-ul
exclude = [
    "Depression_Severity", "Depression_Score", "label",
    "FeelingDownDepressedOrHopeless", "HaveLittleInterestInDoingThings",
    "FeelingBadAboutYourself", "FeelingTiredOrHavingLittleEnergy",
    "MovingOrSpeakingSlowlyOrTooFast", "TroubleConcentratingOnThings",
    "TroubleSleepingOrSleepingTooMuch", "PoorAppetiteOrOvereating",
    "TakeMedicationForDepression", "TakeMedicationForTheseFeelings",
    "HowOftenDoYouFeelDepressed", "HowOftenDoYouFeelWorriedOrAnxious"
]
feature_cols = [c for c in df.columns if c not in exclude]
print(f"Total features: {len(feature_cols)}")

# COMMAND ----------

# 4. Imputare valori lipsa pentru variabilele numerice
# Random Forest in sklearn nu accepta NaN, deci inlocuim valorile lipsa cu mediana
# fiecarei coloane (mai robust decat media, mai putin afectat de valori extreme).
X = df[feature_cols].copy()
for c in X.columns:
    if X[c].isnull().any():
        X[c] = X[c].fillna(X[c].median())

y = df["label"]
print(f"Distributia claselor: {y.value_counts().sort_index().to_dict()}")

# COMMAND ----------

# 5. Split 80% antrenare / 20% testare
# stratify=y mentine proportia claselor in ambele seturi (important pentru
# date dezechilibrate, cum sunt categoriile severe care au putine exemple)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# 6. Antrenare model
print("Training model...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model antrenat!")

# COMMAND ----------

# 7. Evaluare pe setul de test
y_pred = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted")

print(f"\n=== Rezultate Random Forest — Clasificare PHQ-9 ===")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# COMMAND ----------

# 8. Importanta feature-urilor — care variabile contribuie cel mai mult la predictie
feat_imp = pd.DataFrame({
    "Feature":    feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Top 10 Features ===")
print(feat_imp.head(10).to_string(index=False))

# COMMAND ----------

# 9. Salvare model + encoderi pe volum
# Folosim joblib (standard pentru sklearn) si salvam intr-un singur fisier
# atat modelul cat si encoderii — necesari pentru reproducerea encoding-ului
# in notebook-ul de vizualizari.
model_dir = "/Volumes/catalog_licenta/default/volum_licenta"
os.makedirs(model_dir, exist_ok=True)
model_path = f"{model_dir}/rf_depression_model.joblib"

joblib.dump({
    "model":         model,
    "encoders":      encoders,
    "label_encoder": label_encoder,
    "feature_cols":  feature_cols
}, model_path)

print(f"Model salvat la: {model_path}")
