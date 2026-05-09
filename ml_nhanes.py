# Databricks notebook source
# ============================================================
# ML LAYER — Encoding via dense_rank, fara ML cache overhead
# ============================================================

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, dense_rank
from pyspark.sql.window import Window
import pandas as pd

df = spark.table("catalog_licenta.default.gold_nhanes")

# 1. Encoding label Depression_Severity via dense_rank
severity_order = Window.orderBy("Depression_Severity")
df_encoded = df.withColumn("label",
    (dense_rank().over(severity_order) - 1).cast("double"))

# 2. Encoding coloane string via dense_rank - zero ML cache
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

for c in string_cols:
    w = Window.orderBy(c)
    df_encoded = df_encoded \
        .withColumn(f"{c}_Idx",
            (dense_rank().over(w) - 1).cast("double")) \
        .drop(c)

print(f"✅ Encoding complet — {df_encoded.count()} rânduri")

# 3. Feature columns
exclude = [
    "Depression_Severity", "Depression_Score", "label",
    "FeelingDownDepressedOrHopeless", "HaveLittleInterestInDoingThings",
    "FeelingBadAboutYourself", "FeelingTiredOrHavingLittleEnergy",
    "MovingOrSpeakingSlowlyOrTooFast", "TroubleConcentratingOnThings",
    "TroubleSleepingOrSleepingTooMuch", "PoorAppetiteOrOvereating",
    "TakeMedicationForDepression", "TakeMedicationForTheseFeelings",
    "HowOftenDoYouFeelDepressed", "HowOftenDoYouFeelWorriedOrAnxious"
]
feature_cols = [c for c in df_encoded.columns if c not in exclude]
print(f"Total features: {len(feature_cols)}")

# 4. Pipeline minimal — doar Assembler + RandomForest
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    maxDepth=8,
    seed=42
)
pipeline = Pipeline(stages=[assembler, rf])

# 5. Train/Test split 80/20
train_df, test_df = df_encoded.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count()} | Test: {test_df.count()}")

# 6. Antrenare
print("⏳ Training model...")
model = pipeline.fit(train_df)
print("✅ Model antrenat!")

# 7. Evaluare
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)
accuracy  = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1        = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall    = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

print(f"\n=== Rezultate Random Forest — Clasificare PHQ-9 ===")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# 8. Feature Importance
rf_model = model.stages[-1]
feat_imp = pd.DataFrame({
    "Feature":    feature_cols,
    "Importance": rf_model.featureImportances.toArray()
}).sort_values("Importance", ascending=False)

print("\n=== Top 10 Features ===")
print(feat_imp.head(10).to_string(index=False))

# 9. Salvare model
model_path = "/Volumes/catalog_licenta/default/volum_licenta/rf_depression_model"
model.save(model_path)
print(f"\n✅ Model salvat la: {model_path}")

# COMMAND ----------

# ============================================================
# WIDGET INTERACTIV — Depression Severity Predictor
# ============================================================

import ipywidgets as widgets
from IPython.display import display, clear_output
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.functions import col, dense_rank
from pyspark.sql.window import Window
import pandas as pd

# 1. Incarcam modelul salvat
model_path = "/Volumes/catalog_licenta/default/volum_licenta/rf_depression_model"
model = PipelineModel.load(model_path)
print("✅ Model încărcat!")

# 2. Incarcam tabelul Gold pentru mapping-uri
df_gold = spark.table("catalog_licenta.default.gold_nhanes")

# 3. Construim mapping-urile pentru coloanele categorice
# dense_rank ordoneaza alfabetic, deci replicam aceeasi logica
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

mappings = {}
for c in string_cols:
    vals = sorted([
        r[0] for r in df_gold.select(c).distinct().collect()
        if r[0] is not None
    ])
    mappings[c] = {v: float(i) for i, v in enumerate(vals)}

print("✅ Mapping-uri construite!")

# 4. Label mapping (Depression_Severity → label numeric)
severity_vals = sorted([
    r[0] for r in df_gold.select("Depression_Severity").distinct().collect()
    if r[0] is not None
])
label_to_severity = {float(i): v for i, v in enumerate(severity_vals)}
print(f"Clase: {label_to_severity}")

# 5. Construim widget-urile
age        = widgets.IntSlider(value=45, min=18, max=80,
                description="Vârstă (ani):", style={"description_width": "200px"})
sleep      = widgets.FloatSlider(value=7.0, min=3.0, max=12.0, step=0.5,
                description="Ore somn/noapte:", style={"description_width": "200px"})
sedentary  = widgets.IntSlider(value=350, min=0, max=800, step=10,
                description="Minute sedentarism/zi:", style={"description_width": "200px"})
weight     = widgets.FloatSlider(value=80.0, min=40.0, max=150.0, step=0.5,
                description="Greutate (kg):", style={"description_width": "200px"})
waist      = widgets.FloatSlider(value=90.0, min=60.0, max=130.0, step=0.5,
                description="Circumferință talie (cm):", style={"description_width": "200px"})
systolic   = widgets.IntSlider(value=120, min=90, max=180,
                description="Tensiune sistolică (mmHg):", style={"description_width": "200px"})
glyco      = widgets.FloatSlider(value=5.5, min=4.0, max=12.0, step=0.1,
                description="HbA1c (%):", style={"description_width": "200px"})
fiber      = widgets.FloatSlider(value=15.0, min=0.0, max=60.0, step=0.5,
                description="Fibre alimentare (g/zi):", style={"description_width": "200px"})
energy     = widgets.FloatSlider(value=2000.0, min=500.0, max=5000.0, step=50.0,
                description="Energie (kcal/zi):", style={"description_width": "200px"})
alcohol    = widgets.FloatSlider(value=5.0, min=0.0, max=100.0, step=1.0,
                description="Alcool (g/zi):", style={"description_width": "200px"})
trunk_fat  = widgets.FloatSlider(value=12000.0, min=1000.0, max=40000.0, step=500.0,
                description="Grăsime trunchi (g):", style={"description_width": "200px"})
cholest    = widgets.FloatSlider(value=190.0, min=100.0, max=300.0, step=1.0,
                description="Colesterol total (mg/dL):", style={"description_width": "200px"})

# Dropdown-uri pentru categorice
gender_dd     = widgets.Dropdown(options=list(mappings["Gender"].keys()),
                    description="Gen:", style={"description_width": "200px"})
sleepiness_dd = widgets.Dropdown(options=list(mappings["HowOftenFeelOverlySleepyDuringDay"].keys()),
                    description="Somnolență diurnă:", style={"description_width": "200px"})
sleep_trouble = widgets.Dropdown(options=list(mappings["EverToldDoctorHadTroubleSleeping"].keys()),
                    description="Probleme somn (medic):", style={"description_width": "200px"})
income_dd     = widgets.Dropdown(options=list(mappings["AnnualHouseholdIncome"].keys()),
                    description="Venit anual:", style={"description_width": "200px"})
snore_dd      = widgets.Dropdown(options=list(mappings["HowOftenDoYouSnore"].keys()),
                    description="Sforăit:", style={"description_width": "200px"})
marital_dd    = widgets.Dropdown(options=list(mappings["MaritalStatus"].keys()),
                    description="Status marital:", style={"description_width": "200px"})
educ_dd       = widgets.Dropdown(options=list(mappings["EducationLevelAdults20"].keys()),
                    description="Educație:", style={"description_width": "200px"})
smoke_dd      = widgets.Dropdown(options=list(mappings["SmokedAtLeast100CigarettesInLife"].keys()),
                    description="Fumat (>100 țigări):", style={"description_width": "200px"})
walk_dd       = widgets.Dropdown(options=list(mappings["WalkOrBicycle"].keys()),
                    description="Mers/bicicletă:", style={"description_width": "200px"})
walk_diff_dd  = widgets.Dropdown(options=list(mappings["HaveSeriousDifficultyWalking"].keys()),
                    description="Dificultate mers:", style={"description_width": "200px"})
vig_dd        = widgets.Dropdown(options=list(mappings["VigorousRecreationalActivities"].keys()),
                    description="Activitate viguroasă:", style={"description_width": "200px"})
mod_dd        = widgets.Dropdown(options=list(mappings["ModerateRecreationalActivities"].keys()),
                    description="Activitate moderată:", style={"description_width": "200px"})
race_dd       = widgets.Dropdown(options=list(mappings["RacehispanicOrigin"].keys()),
                    description="Origine etnică:", style={"description_width": "200px"})

btn    = widgets.Button(description="🔍 Evaluează Risc",
            button_style="primary", layout=widgets.Layout(width="200px"))
output = widgets.Output()

def on_click(b):
    with output:
        clear_output()
        # Construim randul pacientului
        pacient = {
            "AgeInYearsAtScreening":          float(age.value),
            "MinutesSedentaryActivity":        float(sedentary.value),
            "SleepHoursWeekdaysOrWorkdays":    float(sleep.value),
            "DietaryFiberGm_DR1TOT":           float(fiber.value),
            "EnergyKcal_DR1TOT":               float(energy.value),
            "AlcoholGm_DR1TOT":                float(alcohol.value),
            "WeightKg":                        float(weight.value),
            "WaistCircumferenceCm":            float(waist.value),
            "TrunkFatG":                       float(trunk_fat.value),
            "Glycohemoglobin":                 float(glyco.value),
            "TotalCholesterolMgdl":            float(cholest.value),
            "SystolicBloodPres1StRdgMmHg":     float(systolic.value),
            # Categorice encodate
            "Gender_Idx":                      mappings["Gender"].get(gender_dd.value, 0.0),
            "RacehispanicOrigin_Idx":          mappings["RacehispanicOrigin"].get(race_dd.value, 0.0),
            "MaritalStatus_Idx":               mappings["MaritalStatus"].get(marital_dd.value, 0.0),
            "EducationLevelAdults20_Idx":      mappings["EducationLevelAdults20"].get(educ_dd.value, 0.0),
            "HowOftenDoYouSnore_Idx":          mappings["HowOftenDoYouSnore"].get(snore_dd.value, 0.0),
            "HowOftenFeelOverlySleepyDuringDay_Idx": mappings["HowOftenFeelOverlySleepyDuringDay"].get(sleepiness_dd.value, 0.0),
            "EverToldDoctorHadTroubleSleeping_Idx":  mappings["EverToldDoctorHadTroubleSleeping"].get(sleep_trouble.value, 0.0),
            "SmokedAtLeast100CigarettesInLife_Idx":  mappings["SmokedAtLeast100CigarettesInLife"].get(smoke_dd.value, 0.0),
            "WalkOrBicycle_Idx":               mappings["WalkOrBicycle"].get(walk_dd.value, 0.0),
            "HaveSeriousDifficultyWalking_Idx":mappings["HaveSeriousDifficultyWalking"].get(walk_diff_dd.value, 0.0),
            "VigorousRecreationalActivities_Idx": mappings["VigorousRecreationalActivities"].get(vig_dd.value, 0.0),
            "ModerateRecreationalActivities_Idx": mappings["ModerateRecreationalActivities"].get(mod_dd.value, 0.0),
            "AnnualHouseholdIncome_Idx":       mappings["AnnualHouseholdIncome"].get(income_dd.value, 0.0),
        }

        df_pacient = spark.createDataFrame(pd.DataFrame([pacient]))
        pred = model.transform(df_pacient).select("prediction", "probability").first()
        severitate = label_to_severity.get(pred["prediction"], "Necunoscută")
        prob = max(pred["probability"])

        icons = {
            "Minimala": "🟢", "Usoara": "🟡",
            "Moderata": "🟠", "Moderat_Severa": "🔴", "Severa": "🔴"
        }
        icon = icons.get(severitate, "⚪")

        print(f"\n{'='*45}")
        print(f"  REZULTAT EVALUARE DEPRESIE PHQ-9")
        print(f"{'='*45}")
        print(f"  Severitate:     {icon} {severitate}")
        print(f"  Probabilitate:  {prob*100:.1f}%")
        print(f"{'='*45}")

btn.on_click(on_click)

display(widgets.VBox([
    widgets.HTML("<h3>🧠 Depression Severity Predictor — NHANES 2017-2018</h3>"),
    widgets.HTML("<b>Date demografice și clinice</b>"),
    widgets.HBox([
        widgets.VBox([age, gender_dd, race_dd, marital_dd, educ_dd, income_dd]),
        widgets.VBox([weight, waist, trunk_fat, glyco, cholest, systolic])
    ]),
    widgets.HTML("<b>Stil de viață și somn</b>"),
    widgets.HBox([
        widgets.VBox([sleep, sedentary, fiber, energy, alcohol]),
        widgets.VBox([sleepiness_dd, sleep_trouble, snore_dd,
                      smoke_dd, walk_dd, walk_diff_dd, vig_dd, mod_dd])
    ]),
    btn,
    output
]))
