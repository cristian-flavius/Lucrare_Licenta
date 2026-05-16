# Databricks notebook source
# ==============================================================================
# VIZUALIZĂRI — Grafice statistice bazate pe datele NHANES 2017-2018
# ==============================================================================
# Input:  catalog_licenta.default.gold_nhanes (date agregate)
#
# Acest notebook generează figurile statistice folosite în lucrarea scrisă.
# Modelul Random Forest este reantrenat cu aceiași parametri și seed ca în
# ml_nhanes, pentru a evita incompatibilități de versiune sklearn între clustere.
#
# Grafice generate:
#   1. Distribuția severității depresiei (bar chart)
#   2. Matricea de confuzie a modelului (heatmap)
#   3. Top 10 factori predictivi (bar chart orizontal)
#   4. Biomarkeri clinici per categorie de severitate (grouped bar)
#   5. Distribuția grupelor de vârstă per severitate (stacked bar)
#   6. Relația între sedentarism și severitate (box plot)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# COMMAND ----------

# Configurare globală stil grafice
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.figsize"] = (10, 6)

# Ordinea clinică a categoriilor de severitate PHQ-9
ORDINE_SEVERITATE = ["Minimală", "Ușoară", "Moderată", "Moderat-Severă", "Severă"]

# Paleta de culori: verde (minimală) → roșu (severă)
CULORI_SEVERITATE = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#8e1b1b"]

# Mapping pentru afișarea cu diacritice corecte
ETICHETE_SEVERITATE = {
    "Minimala":       "Minimală",
    "Usoara":         "Ușoară",
    "Moderata":       "Moderată",
    "Moderat_Severa": "Moderat-Severă",
    "Severa":         "Severă"
}

# Director pentru salvarea figurilor
FIG_DIR = "/Volumes/catalog_licenta/default/volum_licenta/figuri"
os.makedirs(FIG_DIR, exist_ok=True)

# COMMAND ----------

# 1. Încărcare date din Gold
df_spark = spark.table("catalog_licenta.default.gold_nhanes")
df = df_spark.toPandas()

# Adăugăm coloana cu diacritice pentru afișare
df["Severitate"] = df["Depression_Severity"].map(ETICHETE_SEVERITATE)
df["Severitate"] = pd.Categorical(df["Severitate"], categories=ORDINE_SEVERITATE, ordered=True)

print(f"Date încărcate: {len(df)} pacienți, {len(df.columns)} coloane")

# COMMAND ----------

# 2. Encoding + antrenare model (identic cu ml_nhanes, aceleași parametri și seed)
# Reantrenăm direct în loc să încărcăm modelul salvat, pentru a evita
# incompatibilități între versiunile scikit-learn ale clusterelor Databricks.

from sklearn.ensemble import RandomForestClassifier

df_ml = df.copy()

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
    df_ml[c] = df_ml[c].fillna("UNKNOWN").astype(str)
    le = LabelEncoder()
    df_ml[f"{c}_Idx"] = le.fit_transform(df_ml[c])
    encoders[c] = le
    df_ml = df_ml.drop(columns=[c])

label_encoder = LabelEncoder()
df_ml["label"] = label_encoder.fit_transform(df_ml["Depression_Severity"])

exclude = [
    "Depression_Severity", "Depression_Score", "label", "Severitate",
    "FeelingDownDepressedOrHopeless", "HaveLittleInterestInDoingThings",
    "FeelingBadAboutYourself", "FeelingTiredOrHavingLittleEnergy",
    "MovingOrSpeakingSlowlyOrTooFast", "TroubleConcentratingOnThings",
    "TroubleSleepingOrSleepingTooMuch", "PoorAppetiteOrOvereating",
    "TakeMedicationForDepression", "TakeMedicationForTheseFeelings",
    "HowOftenDoYouFeelDepressed", "HowOftenDoYouFeelWorriedOrAnxious"
]
feature_cols = [c for c in df_ml.columns if c not in exclude]

X = df_ml[feature_cols].copy()
for c in X.columns:
    if X[c].isnull().any():
        X[c] = X[c].fillna(X[c].median())

y = df_ml["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model antrenat — set de test: {len(X_test)} pacienți — predicții generate")

# COMMAND ----------

# ======================================================================
# FIGURA 1 — Distribuția severității depresiei în eșantionul NHANES
# ======================================================================

contorizare = df.groupby("Severitate", observed=True).size().reset_index(name="Număr pacienți")

fig, ax = plt.subplots(figsize=(10, 6))
barras = ax.bar(
    contorizare["Severitate"],
    contorizare["Număr pacienți"],
    color=CULORI_SEVERITATE,
    edgecolor="white",
    linewidth=0.8
)

# Adăugăm numărul și procentul deasupra fiecărei bare
total = contorizare["Număr pacienți"].sum()
for bar, val in zip(barras, contorizare["Număr pacienți"]):
    pct = val / total * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xlabel("Categoria de severitate (conform pragurilor clinice PHQ-9)", fontsize=11)
ax.set_ylabel("Număr de pacienți", fontsize=11)
ax.set_title("Distribuția severității depresiei în eșantionul NHANES 2017-2018", fontsize=13, fontweight="bold")
ax.set_ylim(0, contorizare["Număr pacienți"].max() * 1.2)
sns.despine()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_distributie_severitate.png", bbox_inches="tight")
plt.show()

# COMMAND ----------

# ======================================================================
# FIGURA 2 — Matricea de confuzie a modelului Random Forest
# ======================================================================

etichete_display = [ETICHETE_SEVERITATE[c] for c in label_encoder.classes_]
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=etichete_display,
    yticklabels=etichete_display,
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Număr de predicții"},
    ax=ax
)
ax.set_xlabel("Categoria prezisă de model", fontsize=11)
ax.set_ylabel("Categoria reală (din datele de test)", fontsize=11)
ax.set_title("Matricea de confuzie — Random Forest Classifier", fontsize=13, fontweight="bold")
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_matrice_confuzie.png", bbox_inches="tight")
plt.show()

# COMMAND ----------

# ======================================================================
# FIGURA 3 — Top 10 factori predictivi (importanța variabilelor)
# ======================================================================

# Mapping din numele tehnice NHANES în etichete descriptive în limba română
ETICHETE_FEATURES = {
    "AgeInYearsAtScreening":                    "Vârstă (ani)",
    "WeightKg":                                 "Greutate (kg)",
    "WaistCircumferenceCm":                     "Circumferință talie (cm)",
    "TrunkFatG":                                "Grăsime trunchi (g)",
    "Glycohemoglobin":                          "Hemoglobină glicozilată (%)",
    "TotalCholesterolMgdl":                     "Colesterol total (mg/dL)",
    "SystolicBloodPres1StRdgMmHg":              "Tensiune arterială sistolică (mmHg)",
    "SleepHoursWeekdaysOrWorkdays":             "Ore de somn pe noapte",
    "MinutesSedentaryActivity":                 "Minute de sedentarism pe zi",
    "DietaryFiberGm_DR1TOT":                    "Fibre alimentare (g/zi)",
    "EnergyKcal_DR1TOT":                        "Aport energetic (kcal/zi)",
    "AlcoholGm_DR1TOT":                         "Consum alcool (g/zi)",
    "Gender_Idx":                               "Gen",
    "RacehispanicOrigin_Idx":                   "Origine etnică",
    "MaritalStatus_Idx":                        "Stare civilă",
    "EducationLevelAdults20_Idx":               "Nivel de educație",
    "AnnualHouseholdIncome_Idx":                "Venit anual al gospodăriei",
    "HowOftenDoYouSnore_Idx":                   "Frecvența sforăitului",
    "HowOftenFeelOverlySleepyDuringDay_Idx":    "Somnolență diurnă",
    "EverToldDoctorHadTroubleSleeping_Idx":     "Probleme de somn (raportate la medic)",
    "SmokedAtLeast100CigarettesInLife_Idx":     "Fumat (>100 țigări în viață)",
    "WalkOrBicycle_Idx":                        "Deplasare pe jos / cu bicicleta",
    "HaveSeriousDifficultyWalking_Idx":         "Dificultate serioasă la mers",
    "VigorousRecreationalActivities_Idx":       "Activitate fizică viguroasă",
    "ModerateRecreationalActivities_Idx":       "Activitate fizică moderată",
}

feat_imp = pd.DataFrame({
    "Feature":    feature_cols,
    "Importanță": model.feature_importances_
}).sort_values("Importanță", ascending=True).tail(10)

# Aplicăm etichetele descriptive
feat_imp["Denumire"] = feat_imp["Feature"].map(ETICHETE_FEATURES).fillna(feat_imp["Feature"])

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(feat_imp["Denumire"], feat_imp["Importanță"], color="#2c3e50", edgecolor="white")

# Adăugăm valorile lângă bare
for bar, val in zip(bars, feat_imp["Importanță"]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", ha="left", va="center", fontsize=9)

ax.set_xlabel("Importanță relativă în modelul Random Forest", fontsize=11)
ax.set_title("Top 10 factori predictivi ai severității depresiei", fontsize=13, fontweight="bold")
ax.set_xlim(0, feat_imp["Importanță"].max() * 1.25)
sns.despine()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_top10_features.png", bbox_inches="tight")
plt.show()

# COMMAND ----------

# ======================================================================
# FIGURA 4 — Biomarkeri clinici per categorie de severitate
# ======================================================================

biomarkeri = df.groupby("Severitate", observed=True).agg(
    HbA1c_mediu=("Glycohemoglobin", "mean"),
    Colesterol_mediu=("TotalCholesterolMgdl", "mean"),
    Tensiune_medie=("SystolicBloodPres1StRdgMmHg", "mean"),
    Greutate_medie=("WeightKg", "mean")
).reset_index()

etichete_bio = {
    "HbA1c_mediu":      "HbA1c\n(%)",
    "Colesterol_mediu":  "Colesterol\n(mg/dL)",
    "Tensiune_medie":    "Tensiune\nsistolică\n(mmHg)",
    "Greutate_medie":    "Greutate\n(kg)"
}

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)

for ax, (col, label) in zip(axes, etichete_bio.items()):
    valori = biomarkeri[col].values
    bars = ax.bar(
        biomarkeri["Severitate"],
        valori,
        color=CULORI_SEVERITATE,
        edgecolor="white"
    )
    # Valoarea deasupra fiecărei bare
    for bar, val in zip(bars, valori):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(valori) * 1.15)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    sns.despine(ax=ax)

fig.suptitle("Biomarkeri clinici medii per categorie de severitate a depresiei",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_biomarkeri_per_severitate.png", bbox_inches="tight")
plt.show()

# COMMAND ----------

# ======================================================================
# FIGURA 5 — Distribuția grupelor de vârstă per severitate
# ======================================================================

# Creare grupe de vârstă
df["Grupa de vârstă"] = pd.cut(
    df["AgeInYearsAtScreening"],
    bins=[17, 29, 44, 59, 100],
    labels=["18–29 ani", "30–44 ani", "45–59 ani", "60+ ani"]
)

crosstab = pd.crosstab(df["Grupa de vârstă"], df["Severitate"], normalize="index") * 100

fig, ax = plt.subplots(figsize=(10, 6))
crosstab.plot(
    kind="bar", stacked=True, ax=ax,
    color=CULORI_SEVERITATE, edgecolor="white", linewidth=0.5
)
ax.set_xlabel("Grupa de vârstă", fontsize=11)
ax.set_ylabel("Procent din grupă (%)", fontsize=11)
ax.set_title("Proporția categoriilor de severitate per grupă de vârstă", fontsize=13, fontweight="bold")
ax.legend(title="Severitate", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_varsta_per_severitate.png", bbox_inches="tight")
plt.show()

# COMMAND ----------

# ======================================================================
# FIGURA 6 — Sedentarism și ore de somn per categorie de severitate
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Box plot — Minute de sedentarism
order = ORDINE_SEVERITATE
sns.boxplot(
    data=df, x="Severitate", y="MinutesSedentaryActivity", order=order,
    palette=CULORI_SEVERITATE, ax=ax1, fliersize=2
)
ax1.set_xlabel("Categoria de severitate", fontsize=11)
ax1.set_ylabel("Minute de sedentarism pe zi", fontsize=11)
ax1.set_title("Sedentarismul per severitate", fontsize=12, fontweight="bold")
ax1.tick_params(axis="x", rotation=25)

# Box plot — Ore de somn
sns.boxplot(
    data=df, x="Severitate", y="SleepHoursWeekdaysOrWorkdays", order=order,
    palette=CULORI_SEVERITATE, ax=ax2, fliersize=2
)
ax2.set_xlabel("Categoria de severitate", fontsize=11)
ax2.set_ylabel("Ore de somn pe noapte", fontsize=11)
ax2.set_title("Orele de somn per severitate", fontsize=12, fontweight="bold")
ax2.tick_params(axis="x", rotation=25)

fig.suptitle("Factori comportamentali asociați cu severitatea depresiei",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_sedentarism_somn.png", bbox_inches="tight")
plt.show()

# COMMAND ----------

# ======================================================================
# EXPORT — Confirmare salvare figuri
# ======================================================================

figuri_salvate = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
print(f"\n=== {len(figuri_salvate)} figuri salvate în {FIG_DIR} ===")
for f in sorted(figuri_salvate):
    print(f"   {f}")
print("\nFigurile au fost salvate in format PNG.")