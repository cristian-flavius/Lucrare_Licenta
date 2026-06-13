# Sistem analitic de procesare a datelor și modelarea biomarkerilor. Studiu de caz privind detectarea riscului de depresie utilizând seturi de date epidemiologice
 
**Lucrare de Licență**
 
| | |
|---|---|
| **Universitatea** | Universitatea Politehnica Timișoara |
| **Facultatea** | Automatică și Calculatoare |
| **Specializarea** | Informatică — Învățământ la Distanță |
| **Sesiunea** | Iulie 2026 |
| **Autor** | Cristian-Flavius MATEICA |
| **Coordonator** | Ș.l.dr.ing. Mihaela CRIȘAN-VIDA |
 
---
 
## Descriere
 
Proiectul implementează un pipeline de procesare și analiză a datelor clinice din studiul **NHANES 2017-2018** (National Health and Nutrition Examination Survey, realizat de CDC — Centers for Disease Control and Prevention), cu scopul de a clasifica severitatea depresiei pe baza chestionarului standardizat **PHQ-9** (Patient Health Questionnaire-9).
 
Arhitectura urmează modelul **Medallion** (Bronze → Silver → Gold), un pattern consacrat în ingineria datelor pe platforma Databricks, iar componenta de Machine Learning folosește un clasificator **Random Forest** din scikit-learn pentru a identifica factorii demografici, clinici și comportamentali asociați cu severitatea depresiei.
 
Eșantionul final cuprinde **5.070 de adulți** (≥ 18 ani), clasificați în 5 categorii de severitate conform pragurilor clinice PHQ-9: Minimală, Ușoară, Moderată, Moderat-Severă și Severă.
 
---
 
## Arhitectură
 
```
┌────────────────────────────────────────────────────────────────────────────┐
│                        NHANES 2017-2018 (CDC)                              │
│                     8.366 subiecți × 197 variabile                         │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  BRONZE LAYER  │  Ingestie date brute → Delta Table                        │
│                │  bronze_nhanes.py                                         │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  SILVER LAYER  │  Filtrare adulți, tratare valori aberante PHQ-9,          │
│                │  calcul Depression_Score → Delta Table                    │
│                │  silver_nhanes.py                                         │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  GOLD LAYER    │  Clasificare severitate (5 categorii PHQ-9),              │
│                │  agregări statistice per categorie → Delta Table          │
│                │  gold_nhanes.py                                           │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  ML LAYER      │  Encoding, antrenare Random Forest, evaluare              │
│                │  ml_nhanes.py                                             │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  VIZUALIZĂRI   │  6 figuri statistice pentru capitolul de Rezultate        │
│                │  visualizations.py                                        │
└────────────────────────────────────────────────────────────────────────────┘
```
 
---
 
## Structura repository-ului
 
| Fișier | Descriere |
|---|---|
| `bronze_nhanes.py` | Descarcă datele NHANES 2017-2018 de pe serverele CDC și le salvează ca Delta Table. |
| `silver_nhanes.py` | Selectează 37 de coloane relevante, filtrează adulții (≥ 18 ani), tratează valorile aberante din chestionarul PHQ-9 și calculează scorul total de depresie. |
| `gold_nhanes.py` | Clasifică pacienții în 5 categorii de severitate conform pragurilor clinice PHQ-9 și generează agregări statistice. |
| `ml_nhanes.py` | Codifică variabilele categorice, antrenează un clasificator Random Forest (scikit-learn), evaluează performanța și salvează modelul. |
| `visualizations.py` | Generează 6 figuri statistice (distribuții, matrice de confuzie, importanța variabilelor, biomarkeri, demografie). |
| `requirements.txt` | Lista dependențelor Python necesare. |
 
---
 
## Tehnologii utilizate
 
| Categorie | Tehnologie |
|---|---|
| Platformă | Databricks Community Edition (Free Edition) |
| Motor de procesare | Apache Spark 3.x (PySpark) |
| Format de stocare | Delta Lake |
| Limbaj | Python 3.11+ |
| Machine Learning | scikit-learn (RandomForestClassifier) |
| Vizualizare | matplotlib, seaborn |
| Analiză date | pandas, NumPy |
| Sursă date | NHANES 2017-2018 (pachetul Python `nhanes`) |
| Versionare cod | Git + GitHub |
 
---
 
## Instrucțiuni de rulare
 
### Cerințe preliminare
 
1. Cont Databricks Community Edition (gratuit): [community.cloud.databricks.com](https://community.cloud.databricks.com)
2. Cluster activ (se creează automat la prima utilizare)
### Pași de execuție
 
Notebook-urile se rulează **strict în această ordine**, fiecare depinzând de tabela creată de cel anterior:
 
```
1. bronze_nhanes.py     →  creează tabela: catalog_licenta.default.bronze_nhanes
2. silver_nhanes.py     →  creează tabela: catalog_licenta.default.silver_nhanes
3. gold_nhanes.py       →  creează tabela: catalog_licenta.default.gold_nhanes
4. ml_nhanes.py         →  antrenează modelul și îl salvează pe volum
5. visualizations.py    →  generează figurile (necesită tabela din pasul 3)
```
 
### Notă privind Spark MLlib
 
Proiectul folosește scikit-learn pentru componenta de Machine Learning deoarece Databricks Free Edition utilizează Spark Connect, care restricționează anumite operații MLlib. Cu un eșantion de 5.070 de înregistrări, scikit-learn este mai eficient decât Spark MLlib, al cărui overhead de inițializare se justifică doar la volume de date semnificativ mai mari.
 
---
 
## Sursa datelor
 
**NHANES** (National Health and Nutrition Examination Survey) este un studiu realizat de **CDC** (Centers for Disease Control and Prevention, SUA) care evaluează starea de sănătate și nutriție a populației americane prin interviuri și examinări fizice.
 
- **Ciclu utilizat:** 2017-2018
- **Acces:** datele sunt publice și descărcate programatic prin pachetul Python [`nhanes`](https://pypi.org/project/nhanes/)
- **Referință oficială:** [https://www.cdc.gov/nchs/nhanes/](https://www.cdc.gov/nchs/nhanes/)
---
 
## Declarație privind utilizarea instrumentelor de Inteligență Artificială
 
În conformitate cu **Hotărârea Senatului UPT nr. 85 / 25.05.2023** privind recomandările de utilizare a instrumentelor de inteligență artificială:
 
- **Instrument utilizat:** Claude (Anthropic) — [claude.ai](https://claude.ai)
- **Scop:** asistență în scrierea și revizuirea codului Python, structurarea pipeline-ului de date, generarea vizualizărilor și organizarea repository-ului
- **Perioadă de utilizare:** martie – iunie 2026
- **Analiză critică:** toate rezultatele generate au fost verificate, testate și adaptate manual. Deciziile metodologice (alegerea variabilelor, pragurile PHQ-9, strategia de excludere a data leakage) au fost fundamentate pe literatura de specialitate și validate prin rularea efectivă a pipeline-ului pe date reale.