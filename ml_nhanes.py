# Databricks notebook source
# ==============================================================================
# ML LAYER — Antrenare model Random Forest pentru clasificarea severitatii depresiei
# ==============================================================================
# Input:  catalog_licenta.default.gold_nhanes (date cu Depression_Severity)
# Output: Model RandomForest salvat pe volum + widget interactiv de inferenta
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
# Salvam encoderii pentru a putea face inferente identice in widget.
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
# atat modelul cat si encoderii — necesari pentru a reproduce exact encoding-ul
# in widget-ul de inferenta.
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

# COMMAND ----------

# ============================================================
# WIDGET INTERACTIV — Depression Severity Predictor
# ============================================================
# Permite introducerea manuala a datelor unui pacient si afiseaza predictia
# severitatii depresiei conform standardului PHQ-9. Folosit ca demonstratie
# vizuala a modelului antrenat.
# ============================================================

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import joblib

# 1. Incarcare model si encoderi
model_path = "/Volumes/catalog_licenta/default/volum_licenta/rf_depression_model.joblib"
artifacts = joblib.load(model_path)
model         = artifacts["model"]
encoders      = artifacts["encoders"]
label_encoder = artifacts["label_encoder"]
feature_cols  = artifacts["feature_cols"]
print("Model incarcat!")

# 2. Construim optiunile dropdown-urilor din encoderii salvati
# (asa garantam ca valorile dropdown-ului sunt exact cele cunoscute de model)
def options_for(col):
    return [v for v in encoders[col].classes_ if v != "UNKNOWN"]

# COMMAND ----------

# 3. Construim widget-urile
# Sliders pentru variabile numerice
age        = widgets.IntSlider(value=45, min=18, max=80,
                description="Varsta (ani):", style={"description_width": "200px"})
sleep      = widgets.FloatSlider(value=7.0, min=3.0, max=12.0, step=0.5,
                description="Ore somn/noapte:", style={"description_width": "200px"})
sedentary  = widgets.IntSlider(value=350, min=0, max=800, step=10,
                description="Minute sedentarism/zi:", style={"description_width": "200px"})
weight     = widgets.FloatSlider(value=80.0, min=40.0, max=150.0, step=0.5,
                description="Greutate (kg):", style={"description_width": "200px"})
waist      = widgets.FloatSlider(value=90.0, min=60.0, max=130.0, step=0.5,
                description="Circumferinta talie (cm):", style={"description_width": "200px"})
systolic   = widgets.IntSlider(value=120, min=90, max=180,
                description="Tensiune sistolica (mmHg):", style={"description_width": "200px"})
glyco      = widgets.FloatSlider(value=5.5, min=4.0, max=12.0, step=0.1,
                description="HbA1c (%):", style={"description_width": "200px"})
fiber      = widgets.FloatSlider(value=15.0, min=0.0, max=60.0, step=0.5,
                description="Fibre alimentare (g/zi):", style={"description_width": "200px"})
energy     = widgets.FloatSlider(value=2000.0, min=500.0, max=5000.0, step=50.0,
                description="Energie (kcal/zi):", style={"description_width": "200px"})
alcohol    = widgets.FloatSlider(value=5.0, min=0.0, max=100.0, step=1.0,
                description="Alcool (g/zi):", style={"description_width": "200px"})
trunk_fat  = widgets.FloatSlider(value=12000.0, min=1000.0, max=40000.0, step=500.0,
                description="Grasime trunchi (g):", style={"description_width": "200px"})
cholest    = widgets.FloatSlider(value=190.0, min=100.0, max=300.0, step=1.0,
                description="Colesterol total (mg/dL):", style={"description_width": "200px"})

# Dropdown-uri pentru variabile categorice
gender_dd     = widgets.Dropdown(options=options_for("Gender"),
                    description="Gen:", style={"description_width": "200px"})
race_dd       = widgets.Dropdown(options=options_for("RacehispanicOrigin"),
                    description="Origine etnica:", style={"description_width": "200px"})
marital_dd    = widgets.Dropdown(options=options_for("MaritalStatus"),
                    description="Status marital:", style={"description_width": "200px"})
educ_dd       = widgets.Dropdown(options=options_for("EducationLevelAdults20"),
                    description="Educatie:", style={"description_width": "200px"})
income_dd     = widgets.Dropdown(options=options_for("AnnualHouseholdIncome"),
                    description="Venit anual:", style={"description_width": "200px"})
snore_dd      = widgets.Dropdown(options=options_for("HowOftenDoYouSnore"),
                    description="Sforait:", style={"description_width": "200px"})
sleepiness_dd = widgets.Dropdown(options=options_for("HowOftenFeelOverlySleepyDuringDay"),
                    description="Somnolenta diurna:", style={"description_width": "200px"})
sleep_trouble = widgets.Dropdown(options=options_for("EverToldDoctorHadTroubleSleeping"),
                    description="Probleme somn (medic):", style={"description_width": "200px"})
smoke_dd      = widgets.Dropdown(options=options_for("SmokedAtLeast100CigarettesInLife"),
                    description="Fumat (>100 tigari):", style={"description_width": "200px"})
walk_dd       = widgets.Dropdown(options=options_for("WalkOrBicycle"),
                    description="Mers/bicicleta:", style={"description_width": "200px"})
walk_diff_dd  = widgets.Dropdown(options=options_for("HaveSeriousDifficultyWalking"),
                    description="Dificultate mers:", style={"description_width": "200px"})
vig_dd        = widgets.Dropdown(options=options_for("VigorousRecreationalActivities"),
                    description="Activitate viguroasa:", style={"description_width": "200px"})
mod_dd        = widgets.Dropdown(options=options_for("ModerateRecreationalActivities"),
                    description="Activitate moderata:", style={"description_width": "200px"})

btn    = widgets.Button(description="Evalueaza Risc",
            button_style="primary", layout=widgets.Layout(width="200px"))
output = widgets.Output()

# COMMAND ----------

# 4. Logica de predictie

# Mapping intre dropdown-uri si numele coloanelor pentru encoding
categorical_widgets = {
    "Gender":                              gender_dd,
    "RacehispanicOrigin":                  race_dd,
    "MaritalStatus":                       marital_dd,
    "EducationLevelAdults20":              educ_dd,
    "AnnualHouseholdIncome":               income_dd,
    "HowOftenDoYouSnore":                  snore_dd,
    "HowOftenFeelOverlySleepyDuringDay":   sleepiness_dd,
    "EverToldDoctorHadTroubleSleeping":    sleep_trouble,
    "SmokedAtLeast100CigarettesInLife":    smoke_dd,
    "WalkOrBicycle":                       walk_dd,
    "HaveSeriousDifficultyWalking":        walk_diff_dd,
    "VigorousRecreationalActivities":      vig_dd,
    "ModerateRecreationalActivities":      mod_dd,
}

def on_click(b):
    with output:
        clear_output()

        # Construim valorile feature-urilor pentru pacient
        pacient = {
            "AgeInYearsAtScreening":          float(age.value),
            "MinutesSedentaryActivity":       float(sedentary.value),
            "SleepHoursWeekdaysOrWorkdays":   float(sleep.value),
            "DietaryFiberGm_DR1TOT":          float(fiber.value),
            "EnergyKcal_DR1TOT":              float(energy.value),
            "AlcoholGm_DR1TOT":               float(alcohol.value),
            "WeightKg":                       float(weight.value),
            "WaistCircumferenceCm":           float(waist.value),
            "TrunkFatG":                      float(trunk_fat.value),
            "Glycohemoglobin":                float(glyco.value),
            "TotalCholesterolMgdl":           float(cholest.value),
            "SystolicBloodPres1StRdgMmHg":    float(systolic.value),
        }

        # Encoding categorice via encoderii salvati
        for col_name, widget_obj in categorical_widgets.items():
            le = encoders[col_name]
            pacient[f"{col_name}_Idx"] = int(le.transform([widget_obj.value])[0])

        # Construim DataFrame in ordinea exacta a feature-urilor folosite la training
        df_pacient = pd.DataFrame([pacient])[feature_cols]

        # Predictie
        pred_label = model.predict(df_pacient)[0]
        pred_proba = model.predict_proba(df_pacient)[0]
        severitate = label_encoder.inverse_transform([pred_label])[0]
        prob       = max(pred_proba)

        icons = {
            "Minimala":       "🟢",
            "Usoara":         "🟡",
            "Moderata":       "🟠",
            "Moderat_Severa": "🔴",
            "Severa":         "🔴"
        }
        icon = icons.get(severitate, "⚪")

        print(f"\n{'='*45}")
        print(f"  REZULTAT EVALUARE DEPRESIE PHQ-9")
        print(f"{'='*45}")
        print(f"  Severitate:     {icon} {severitate}")
        print(f"  Probabilitate:  {prob*100:.1f}%")
        print(f"{'='*45}")

btn.on_click(on_click)

# COMMAND ----------

# 5. Afisare interfata
display(widgets.VBox([
    widgets.HTML("<h3>Depression Severity Predictor — NHANES 2017-2018</h3>"),
    widgets.HTML("<b>Date demografice si clinice</b>"),
    widgets.HBox([
        widgets.VBox([age, gender_dd, race_dd, marital_dd, educ_dd, income_dd]),
        widgets.VBox([weight, waist, trunk_fat, glyco, cholest, systolic])
    ]),
    widgets.HTML("<b>Stil de viata si somn</b>"),
    widgets.HBox([
        widgets.VBox([sleep, sedentary, fiber, energy, alcohol]),
        widgets.VBox([sleepiness_dd, sleep_trouble, snore_dd,
                      smoke_dd, walk_dd, walk_diff_dd, vig_dd, mod_dd])
    ]),
    btn,
    output
]))
