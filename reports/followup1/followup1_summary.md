# Follow-up #1 Deliverable — Obtain, Scrub & Explore (Dec 4)

**Project:** Predicting Formula 1 Points Finishes (2010–2024): A Temporal Ablation of Qualifying Position, Rolling Constructor Momentum, and Historical Circuit Affinity  
**Course:** Fundamentals of Machine Learning (2026–27) — UAB  
**Team 05:** Marc Font Hostalet, David Redrejo Pérez, Elias Barreiro Pena  
**Primary Executable Artifact:** [`notebooks/01_followup1_obtain_scrub_explore.ipynb`](../../notebooks/01_followup1_obtain_scrub_explore.ipynb)  
**Processed Handoff Dataset:** [`data/processed/f1_modeling_2010_2024.parquet`](../../data/processed/f1_modeling_2010_2024.parquet)

---

## 1. Instructor Feedback Compliance Matrix

| Instructor Feedback Point | Implemented Engineering Decision | Verified Evidence |
| :--- | :--- | :--- |
| Restrict period (scoring rules changed) | Target modeling restricted to **2010–2024** (unified FIA Top-10 points rule); 1995–2009 retained *only* for strictly past warm-start history | **6,436** entries, **305** races, **47.39%** Top-10 rate (`0` discrepancies vs `points > 0`) |
| Never use in-race variables | Excluded `lap_times`, `pit_stops`, target-race `status`, race `time`/`laps`/`fastestLap`, and post-race standings | `FORBIDDEN_INRACE_COLUMNS` intersection = **0** |
| Rolling features strictly from prior races | Race-level aggregation + strict `.shift(1)` before rolling/expanding statistics (blocks both same-race and same-race teammate leakage) | Adversarial future-mutation & same-race-mutation `max_abs_diff = 0.0` |
| Split strictly by season | **Dev:** 2010–2021 (12 seasons, $n=5{,}077$; 5-fold expanding-window CV on 2017–2021); **Test:** 2022–2024 Ground-Effect era ($n=1{,}359$) | All EDA feature-target statistics computed strictly on 2010–2021 Dev |
| `grid <= 10` (~77%) baseline | Evaluated naive raw (`76.03%`) vs scrubbed `1 <= grid <= 10` (`76.91%` overall; `76.86%` Dev; `77.12%` Test) | Scrubbing `grid == 0` pit-lane starts lifts baseline **+0.89 pp** |
| Three-group ablation setup | 26 pre-race features partitioned into disjoint Groups **A (8)**, **B (11)**, and **C (7)** | Ready for full $2^3 - 1 = 7$-subset ablation in Follow-up #2 |

---

## 2. Key Scrubbing & Feature Engineering Wins

1. **Pit-Lane Starts (`grid == 0`, 85 entries):** Raw Ergast encodes pit-lane starts and DNS entries as `grid = 0`, which naive `grid <= 10` falsely classifies as pole position. Mapping `grid == 0` to `n_entries + 1` with an explicit pre-race flag `is_pit_lane_start` corrects the artifact (+0.89 pp baseline accuracy).
2. **Canonical Constructor Lineages (`23 raw IDs -> 13 lineages`):** Rebrands break longitudinal team continuity—notably Ergast maps Kick Sauber (2024) back to `constructorId = 15` (last used in 2018) after 5 years as Alfa Romeo (ID 51). Canonicalizing team lineages (`Sauber <-> Alfa Romeo`, `Toro Rosso -> AlphaTauri -> RB`, `Force India -> Racing Point -> Aston Martin`, `Renault/Lotus -> Alpine`) preserves continuous rolling team form and circuit history.
3. **Bayesian / MAP Smoothing & Non-Collinear Grid Penalties:** Small-sample driver/constructor circuit rates and early rolling rates use Empirical Bayes shrinkage toward neutral fixed domain priors (Course Lectures 5–6). Defining `grid_penalty_drop = max(0, grid_clean - qualy_pos)` isolates engine/gearbox penalty drops while avoiding exact linear rank deficiency.

---

## 3. Exploratory Data Analysis Highlights (Dev 2010–2021, $n=5{,}077$)

* **Where the `grid <= 10` Baseline Fails:** Outside P8–P13 (72.1% of entries), `grid <= 10` reaches **83.55%** accuracy; inside the **P8–P13 midfield transition zone** (27.9% of entries), accuracy collapses to **59.59%**. Furthermore, top-tier constructors starting P11+ still recover into the Top-10 **55%–59%** of the time (and Q5 top teams starting P16+/Pit finish Top-10 **66%** of the time vs **2%** for Q1 backmarkers).
* **Early Signal Preview Across Factor Groups:**
  * **Group A (Qualifying & Grid, 8 features):** Max univariate ROC-AUC **0.842** (`qualy_pos`, edging out `grid_clean` at 0.836), mean MI **0.118 nats**.
  * **Group B (Rolling Form, 11 features):** Max univariate ROC-AUC **0.835** (`const_season_pts_per_race`; `const_roll10_avg_pos` 0.833; `const_roll5_pts_per_car` 0.833), highest group mean AUC **0.728** and mean MI **0.128 nats**.
  * **Group C (Circuit History, 7 features):** Max univariate ROC-AUC **0.704** (`const_circuit_hist_top10_rate`), mean AUC **0.576**, mean MI **0.029 nats**—strongly foreshadowing Hypothesis H1.
* **Multicollinearity Diagnosis:** High within-group VIFs (`grid_clean`: 139.9, `qualy_pos`: 104.1, `grid_pct`: 53.5, constructor rolling metrics: 13.8–22.2) directly motivate **L2 (Ridge) and L1 (Lasso) regularization** in Follow-up #2.

---

## 4. Follow-up #1 Visual Artifacts (`reports/figures/followup1/`)

1. `fig01_scoring_eras_and_temporal_split.png` — FIA scoring regimes (1950–2024) & strict seasonal train/CV/test split.
2. `fig02_grid_vs_top10_and_pitlane_scrub.png` — $P(\text{Top-10} \mid \text{Grid})$ with 95% Wilson CIs, pit-lane scrub & P11+ recovery rates.
3. `fig03_factor_group_signal_strength.png` — Univariate ROC-AUC & Mutual Information across all 26 features by factor group.
4. `fig04_constructor_momentum_and_lineage.png` — Grid $\times$ Constructor Form interaction heatmap & canonical lineage trajectories.
5. `fig05_circuit_overtaking_vs_attrition.png` — Circuit overtaking difficulty vs historical attrition regimes across 21 tracks.
6. `fig06_correlation_matrix_and_multicollinearity.png` — Group-ordered Spearman correlation heatmap & Top-12 VIFs.

---

## 5. Next Steps for Follow-up #2 (Dec 11)

Proceed to [`notebooks/02_followup2_model_interpret.ipynb`](../../notebooks/02_followup2_model_interpret.ipynb) to benchmark 9 course ML models vs 3 baselines under 5-fold expanding-window seasonal CV, execute the 7-subset ablation study with exact Shapley attributions, perform L1/L2 regularization & rolling-window what-if studies, and conduct post-hoc irreducible DNF error decomposition on the 2022–2024 Ground-Effect Test set.
