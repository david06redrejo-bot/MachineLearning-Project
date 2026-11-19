# Follow-up #2 Deliverable — Model & iNterpret (Dec 11)

**Project:** Predicting Formula 1 Points Finishes (2010–2024): A Temporal Ablation of Qualifying Position, Rolling Constructor Momentum, and Historical Circuit Affinity  
**Course:** Fundamentals of Machine Learning (2026–27) — UAB  
**Team 05:** Marc Font Hostalet, David Redrejo Pérez, Elias Barreiro Pena  
**Primary Executable Artifact:** [`notebooks/02_followup2_model_interpret.ipynb`](../../notebooks/02_followup2_model_interpret.ipynb)  
**Serialized Models:** `models/*.joblib` | **Experiment Metrics:** `reports/metrics/followup2_*.csv/.json`

---

## 1. Benchmark Summary (5-Fold Expanding-Window CV 2017–2021 & Held-Out Test 2022–2024, $n=1{,}359$)

| Model | Family | CV ROC-AUC | CV Acc (%) | Test ROC-AUC | Test Acc (%) | Test F1 | Test Log-Loss | Test Brier | McNemar $p$ vs `grid<=10` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: Majority Class | Baseline | 0.5000 | 50.00 | 0.5000 | 49.96 | 0.0000 | 6.9128 | 0.5004 | — |
| **Baseline: `grid <= 10` (Scrubbed)** | Baseline | 0.7520 | 75.20 | 0.7712 | **77.12** | 0.7708 | — | — | *(ref)* |
| Baseline: Grid-Only Logistic | Baseline | 0.8186 | 75.20 | 0.8316 | 77.12 | 0.7708 | 0.5077 | 0.1651 | — |
| **Logistic Regression (L2, $C=0.1$)** | Linear | 0.8387 | 76.67 | 0.8708 | **80.43** | 0.7997 | 0.4522 | 0.1437 | **0.0012** |
| **Logistic Regression (L1, $C=0.01$)** | Linear | **0.8401** | **76.90** | 0.8728 | 79.84 | 0.7964 | 0.4579 | 0.1450 | **0.0087** |
| **Linear SVM ($C=0.001$, Platt)** | Linear | 0.8376 | 76.23 | 0.8717 | 79.76 | 0.7943 | **0.4504** | **0.1428** | **0.0061** |
| Gaussian Naive Bayes | Probabilistic | 0.8323 | 76.39 | 0.8609 | 80.28 | 0.8027 | 1.5255 | 0.1810 | 0.0017 |
| k-Nearest Neighbors ($k=75$) | Instance-Based | 0.8290 | 75.17 | 0.8671 | 80.06 | 0.8018 | 0.4613 | 0.1467 | 0.0012 |
| RBF Kernel SVM | Non-Linear Kernel | 0.8365 | 75.94 | 0.8686 | 80.13 | 0.8038 | 0.4524 | 0.1432 | <0.0001 |
| Decision Tree (Entropy, depth 4) | Tree | 0.8201 | 74.65 | 0.8554 | 78.59 | 0.7846 | 0.4764 | 0.1513 | 0.2475 |
| **Random Forest** | Tree Ensemble | **0.8401** | 76.19 | **0.8734** | 79.91 | 0.8020 | 0.4531 | 0.1438 | **0.0062** |
| **Gradient Boosting** | Tree Ensemble | 0.8393 | 76.10 | 0.8720 | **81.16** | **0.8153** | 0.4534 | 0.1440 | **<0.0001** |

---

## 2. Centerpiece Result: 7-Subset Factor Group Ablation & Exact Shapley Attribution

| Feature Subset | # Feat. | LogReg (L2) AUC / Acc | Linear SVM AUC / Acc | RBF SVM AUC / Acc | Random Forest AUC / Acc | Grad. Boost AUC / Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `C` (Track only) | 7 | 0.778 / 72.3% | 0.778 / 72.0% | 0.742 / 69.5% | 0.750 / 69.5% | 0.750 / 69.1% |
| `B` (Rolling Form only) | 11 | 0.837 / 77.6% | 0.835 / 77.5% | 0.834 / 77.6% | 0.833 / 77.5% | 0.834 / 77.3% |
| `A` (Qualifying only) | 8 | 0.849 / 78.4% | 0.850 / 78.7% | 0.848 / 78.4% | 0.849 / 77.8% | 0.850 / 77.2% |
| `B+C` (Form + Track) | 18 | 0.837 / 77.6% | 0.835 / 77.9% | 0.832 / 77.5% | 0.834 / 77.4% | 0.834 / 77.4% |
| `A+C` (Qualifying + Track) | 15 | 0.858 / 79.1% | 0.860 / 78.7% | 0.857 / 78.4% | 0.857 / 78.3% | 0.859 / 78.9% |
| **`A+B` (Qualifying + Form)** | **19** | **0.871 / 80.0%** | **0.871 / 80.1%** | **0.867 / 80.1%** | **0.871 / 79.9%** | **0.871 / 80.9%** |
| `A+B+C` (Full Set) | 26 | 0.871 / 80.4% | 0.872 / 79.8% | 0.869 / 80.1% | 0.873 / 79.9% | 0.872 / 81.2% |

* **Confirmation of Hypothesis H1:** Adding Group B (`A -> A+B`) delivers **+0.019 to +0.022 Test ROC-AUC** universally across all 5 model families. By contrast, adding Group C (`A+B -> A+B+C`) contributes **-0.0002 to +0.0025 AUC**, and CV-optimal **L1 Lasso ($C=0.01$)** retains **10/26 features: 4 from Group A, 6 from Group B, and 0/7 from Group C!**
* **Shapley Insight:** Across models, Groups A and B capture **74.6%–77.6%** of ROC-AUC Shapley gain and **82.6%–86.0%** of Log-Loss reduction; standalone Group C achieves ~0.75–0.78 AUC only because historical circuit top-10 rates serve as a lagged proxy for persistent team quality when A and B are omitted.

---

## 3. Critical Appraisal: Slice Gains, Calibration & Irreducible Race-Day Noise

* **Hard Slices Where ML Shines (Logistic Regression L2 vs `grid <= 10`):**
  * **Grid Penalty Drops ($\ge 3$ places, $n=83$):** `55.4% -> 79.5%` (**+24.1 pp**, AUC 0.873) — e.g., Verstappen 2024 Spa (Grid P11, Q1 $\rightarrow$ $p=86.9\%$ $\rightarrow$ P4) and 2022 Spa (Grid P14, Q1 $\rightarrow$ $p=82.1\%$ $\rightarrow$ P1), while correctly demoting Latifi 2022 Monza (promoted to Grid P10 from Q16 $\rightarrow$ $p=13.5\%$ $\rightarrow$ P15).
  * **Midfield Transition Zone (P8–P13, $n=405$):** `61.2% -> 68.1%` (**+6.9 pp**).
  * **By Season:** 2022 **+5.0 pp** (`73.2% -> 78.2%`), 2023 **+5.7 pp** (`73.9% -> 79.5%`), 2024 `-0.4 pp` (`83.7% -> 83.3%`, yet **0.910 ROC-AUC**, reflecting 2024's record-low 11.3% DNF rate).
* **Irreducible Bayes Error Ceiling:** **60.7%** (71/117) of all False Positives stem from unpredictable race-day retirements (30 Crash/Incident DNFs, 36 Mechanical DNFs, 5 DNS/DSQ), including pole-sitters retiring with brake/turbo/power-unit failures (Verstappen Australia 2024, Leclerc Baku & Spain 2022, Sainz COTA 2022). **46.3%** (69/149) of False Negatives occur in high-attrition races with $\ge 2$ Top-10 starters retiring.
* **What-If Takeaways:** Longer rolling windows ($W=10$ races: Test AUC **0.8751**) dampen per-race DNF variance; canonical constructor lineages improve rebranded team log-loss (`0.5152 -> 0.5132`); CV-selected L1 threshold (`t=0.455`) lifts L1 accuracy `79.84% -> 80.35%` and F1 `0.796 -> 0.810`.
