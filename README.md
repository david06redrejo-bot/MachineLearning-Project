# Predicting Formula 1 Points Finishes (2010–2024)
### A Temporal Ablation of Qualifying Position, Rolling Constructor Momentum, and Historical Circuit Affinity

* **Course:** Fundamentals of Machine Learning (2026–27) — Autonomous University of Barcelona (UAB)
* **Project Team 05:** Marc Font Hostalet, David Redrejo Pérez, Elias Barreiro Pena
* **Dataset:** Formula 1 World Championship (1950–2024) — Ergast / Kaggle (`rohanrao/formula-1-world-championship-1950-2020`)
* **Workflow:** **Obtain → Scrub → Explore** (**Follow-up #1, Dec 4**) | **Model → iNterpret** (**Follow-up #2, Dec 11**) | **Final Report & Presentation** (**Dec 16 & 18**)

---

## Project Overview & Research Hypothesis

We formulate Formula 1 points-finish prediction (`top10 = (positionOrder <= 10)`) across the modern Top-10 scoring era (**2010–2024**, 305 Grands Prix, **6,436** driver-race entries, **47.39%** balanced positive rate) using **26 strictly pre-race features** partitioned into three disjoint factor groups:

1. **Group A — Qualifying & Grid Context (8 features):** scrubbed starting grid (`grid == 0` pit-lane starts remapped to back of grid), normalized grid percentile, pit-lane indicator, qualifying rank, non-negative grid penalty drop, Q3 indicator, scale-invariant qualifying gap to pole (%), and qualifying gap vs teammate (%).
2. **Group B — Rolling Constructor & Driver Momentum (11 features):** strictly prior-race rolling team and driver performance, momentum delta (3- vs 10-race pace), mechanical reliability history, current-season pace entering the round, racecraft (positions gained), and career experience—computed over canonicalized **constructor lineages** so team rebrands (e.g. *Sauber ↔ Alfa Romeo ↔ Kick Sauber*, *Toro Rosso → AlphaTauri → RB*, *Force India → Racing Point → Aston Martin*) retain continuity.
3. **Group C — Historical Track & Circuit Characteristics (7 features):** circuit altitude, strictly prior historical grid-to-finish Spearman correlation (overtaking difficulty), historical circuit attrition/DNF rate, P1–P10 retention rate, and Empirical-Bayes-smoothed driver & constructor circuit affinity.

**Methodological Guardrails:**
* **Zero In-Race Leakage:** All in-race/post-race variables (`lap_times`, `pit_stops`, target-race `status`, race `time`/`laps`/`fastestLap`, post-race standings) are strictly excluded and verified by automated schema checks.
* **Strictly Past-Only Rolling & Circuit Features:** Aggregated to race level and shifted strictly by 1 race (preventing both target leakage and same-race teammate leakage), verified by **adversarial future-mutation and same-race-mutation invariance tests** (`max_abs_diff == 0.0`).
* **Strict Season-Based Temporal Split:**
  * **Development Set:** 2010–2021 (12 seasons, $n = 5{,}077$), with **5-fold expanding-window seasonal CV** on validation seasons 2017, 2018, 2019, 2020, 2021.
  * **Held-Out Test Set:** 2022–2024 Ground-Effect era (3 seasons, $n = 1{,}359$).
* **Strong Domain Baseline:** Scrubbed `1 <= grid <= 10` heuristic (~77% accuracy), evaluated alongside Majority Dummy and Single-Feature Grid Logistic baselines.

---

## Separate Deliveries & Repository Structure

```text
.
├── data/
│   ├── raw/                                        <- 14 Ergast/Kaggle CSV tables (auto-downloaded if absent)
│   └── processed/                                  <- Scrubbed 2010-2024 modeling dataset & audit JSONs
├── notebooks/
│   ├── 01_followup1_obtain_scrub_explore.ipynb     <- DELIVERY 1 (Follow-up #1 — Dec 4): Obtain, Scrub, Features, Leakage Audit, EDA
│   └── 02_followup2_model_interpret.ipynb          <- DELIVERY 2 (Follow-up #2 — Dec 11): Modeling, Temporal CV, 7-Subset Ablation, What-If & Interpret
├── src/
│   ├── data.py                                     <- Obtain + Scrub (grid==0 fix, lineage canonicalization, status taxonomy)
│   ├── features.py                                 <- Leakage-free 3-group feature engineering & automated adversarial leakage audit
│   ├── eda.py                                      <- Follow-up #1 statistical summaries & 6 publication figures
│   ├── models.py                                   <- Follow-up #2 model suite, temporal CV, 7-subset ablation, Shapley & what-if studies
│   ├── evaluation.py                               <- Follow-up #2 interpretability, error decomposition, case studies & 6 figures
│   └── utils.py                                    <- Shared paths, Okabe-Ito publication theme, logging, JSON helpers
├── models/                                         <- Serialized fitted scikit-learn pipelines (.joblib)
├── reports/
│   ├── followup1/
│   │   ├── followup1_summary.md                    <- Delivery 1 concise brief (Markdown)
│   │   └── followup1_summary.pdf                   <- Delivery 1 2-page visual brief (LaTeX PDF)
│   ├── followup2/
│   │   ├── followup2_results_ablation.md           <- Delivery 2 results & ablation brief (Markdown)
│   │   └── followup2_results_ablation.pdf          <- Delivery 2 2-page visual brief (LaTeX PDF)
│   ├── figures/
│   │   ├── followup1/                              <- Figures 1–6 (Follow-up #1 EDA & diagnostics)
│   │   └── followup2/                              <- Figures 7–12 (Follow-up #2 modeling, ablation, interpretability & errors)
│   ├── metrics/                                    <- Structured CSV & JSON experiment results
│   ├── final_report.pdf                            <- Final Scientific Report (strictly <= 5 pages, LaTeX PDF + .tex source)
│   ├── final_presentation.html                     <- Interactive Animated Presentation (Pit-Wall Editorial, speaker notes `N`, edit `E`)
│   ├── final_presentation.pdf                      <- Final Oral Presentation Slides (14-slide 16:9 widescreen PDF export)
│   └── autoevaluation_template.md                  <- Team 05 self- & peer-evaluation template
├── run_all.py                                      <- Single-command CLI runner for Follow-up #1, Follow-up #2, or full pipeline
├── requirements.txt                                <- Reproducible Python dependencies
└── README.md
```

---

## Quick Reproduction Instructions

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-f1-project --display-name "Python 3 (.venv F1 ML)"
```

### 2. Run Each Delivery Separately (or End-to-End)
```powershell
# Delivery 1 — Follow-up #1 (Obtain, Scrub, Features, Leakage Audit & EDA)
python run_all.py --stage followup1

# Delivery 2 — Follow-up #2 (Modeling, Temporal CV, 7-Subset Ablation & Evaluation)
python run_all.py --stage followup2

# Or run the entire project end-to-end
python run_all.py --stage all
```

### 3. Open the Executed Delivery Notebooks
* **Follow-up #1 (Dec 4):** `notebooks/01_followup1_obtain_scrub_explore.ipynb`
* **Follow-up #2 (Dec 11):** `notebooks/02_followup2_model_interpret.ipynb`
