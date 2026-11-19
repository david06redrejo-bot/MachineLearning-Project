# Project Autoevaluation & Peer Contribution Template (10% of Project Grade)

**Course:** Fundamentals of Machine Learning (2026–27) — Autonomous University of Barcelona (UAB)  
**Project Team 05:**
- Marc Font Hostalet
- David Redrejo Pérez
- Elias Barreiro Pena

**Project Title:** Predicting Formula 1 Points Finishes (2010–2024): A Temporal Ablation of Qualifying Position, Rolling Constructor Momentum, and Historical Circuit Affinity

> *Instructions: Fill in and sign off as a group before uploading the final submission to Campus Virtual. Be honest, concrete, and reflective—evaluators value thoughtful self-appraisal over generic praise.*

---

## 1. Contribution Distribution Matrix (Suggested Breakdown — Edit to Match Actual Split)

| Workstream / Deliverable | Marc Font Hostalet | David Redrejo Pérez | Elias Barreiro Pena |
| :--- | :---: | :---: | :---: |
| Proposal formulation & literature/domain review | 33% | 34% | 33% |
| **Follow-up #1:** Data acquisition, relational scrubbing (`grid==0`, lineages) | _[e.g., 40%]_ | _[e.g., 35%]_ | _[e.g., 25%]_ |
| **Follow-up #1:** Temporal feature engineering & zero-leakage audit | _[e.g., 30%]_ | _[e.g., 40%]_ | _[e.g., 30%]_ |
| **Follow-up #1:** EDA visualizations & statistical diagnostics | _[e.g., 35%]_ | _[e.g., 30%]_ | _[e.g., 35%]_ |
| **Follow-up #2:** Model suite, expanding-window CV & hyperparameter tuning | _[e.g., 30%]_ | _[e.g., 35%]_ | _[e.g., 35%]_ |
| **Follow-up #2:** 7-subset ablation study, Shapley attribution & what-if tests | _[e.g., 35%]_ | _[e.g., 35%]_ | _[e.g., 30%]_ |
| **Follow-up #2:** Interpretability, error decomposition & case studies | _[e.g., 35%]_ | _[e.g., 30%]_ | _[e.g., 35%]_ |
| Final 5-page LaTeX report & Beamer presentation slides | _[e.g., 33%]_ | _[e.g., 34%]_ | _[e.g., 33%]_ |
| **Overall Share (must sum to 100%)** | **33.3%** | **33.3%** | **33.3%** |

---

## 2. Reflection on Group Dynamics & Technical Lessons Learned

1. **What worked well in our collaboration:**
   - Structuring the repository into modular Python scripts (`src/data.py`, `src/features.py`, `src/eda.py`, `src/models.py`, `src/evaluation.py`) allowed parallel work across Follow-up #1 and Follow-up #2 without notebook merge conflicts.
   - Separating Delivery 1 (Obtain/Scrub/Explore) and Delivery 2 (Model/Interpret) forced us to lock down data hygiene and adversarial leakage verification before touching any held-out test metrics.

2. **Main technical challenges & how we resolved them:**
   - Preventing subtle temporal leakage (same-race teammate leakage in constructor rolling averages and Ergast's `constructorId = 15` reversion for Kick Sauber in 2024) required aggregating to race level before shifting and implementing automated mutation-invariance assertions.
   - Severe within-group collinearity (`VIF > 100` across qualifying/grid features) required pairing standardized pipelines with L2/L1 regularization tuning under expanding-window seasonal CV.

3. **What we would improve with additional time:**
   - Incorporate pre-race weather forecasts (wet vs dry race-day probability), FIA tyre compound allocations, and practice (FP2 long-run) telemetry from FastF1 to narrow the remaining classified pace-surprise gap.
