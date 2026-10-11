# Motor Insurance Claim Prediction · Predictive Analytics
This is a machine learning project on a book of 58,592 motor policies. It predicts which policies will file a claim from vehicle, policyholder and location data. We benchmark Logistic Regression, Decision Tree and Random Forest on imbalance-aware metrics, then segment the book into four risk groups for underwriting and pricing decisions.

---

A machine-learning approach to identifying, explaining and pricing motor-insurance risk. The project predicts whether a motor policy will file a claim (`is_claim`) using vehicle, policyholder and location attributes available at underwriting time.

## Project at a glance

| | |
|---|---|
| **Problem** | Binary classification of `is_claim` on a book of 58,592 policies |
| **Challenge** | Severe class imbalance: 93.6% no-claim vs 6.4% claim, so accuracy is misleading |
| **Models compared** | Logistic Regression, Decision Tree, Random Forest (with no balancing, class weights, and SMOTE) |
| **Metrics** | Precision, Recall, F1, ROC-AUC (not raw accuracy) |
| **Recommended model** | Random Forest + SMOTE (F1 0.165, ROC-AUC 0.644) |
| **Extra analysis** | K-Means segmentation (K=4) into risk-based policyholder groups |

---

## Repository contents

```
.
├── README.md
├── BADM_Final_Project_Loaded_and_Inspected_File.ipynb   # Analysis notebook (Step 1 complete)
├── train.csv                                            # Labelled data: 58,592 rows, 44 columns
├── test.csv                                             # Unlabelled; set aside (see note below)
└── Motor_Insurance_Claim_Prediction.pptx                # Final presentation deck
```

---

## Quick start

### 1. Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Fix the file paths (important)

The notebook's first code cell currently reads:

```python
train = pd.read_csv('/content/train.csv')   # Google Colab path
test  = pd.read_csv('data/test.csv')        # relative path
```

Change these to wherever your files live. For example, if both CSVs sit next to the notebook:

```python
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
```

On Google Colab, upload `train.csv` (and `test.csv`) to the session and use `/content/train.csv` for both.

If you don't have `test.csv`, the cell will raise `FileNotFoundError`. The rest of the analysis only uses `train.csv`, so you can remove the `test` lines along with the notebook cells that reference `test` (the train/test column and ID checks).

### 3. Run

Open the notebook and use **Run all** (Jupyter, VS Code or Colab).

---

## Data

`train.csv`: one row per policy, 44 columns.

| Group | Examples |
|---|---|
| Identifier (dropped) | `policy_id` |
| Policy / policyholder | `policy_tenure`, `age_of_policyholder` |
| Vehicle | `age_of_car`, `make`, `segment`, `model`, `fuel_type`, `engine_type`, `displacement`, `cylinder`, `gross_weight`, `ncap_rating`, `airbags` |
| Text-encoded numerics | `max_power` (e.g. `91bhp@6000rpm`), `max_torque` (e.g. `91Nm@4250rpm`) |
| Yes/No equipment flags | `is_esc`, `is_tpms`, `is_parking_sensors`, `is_brake_assist`, ... |
| Location | `area_cluster`, `population_density` |
| **Target** | `is_claim` (0 = no claim, 1 = claim) |

Data quality (verified in Step 1): no missing values, `policy_id` unique, no train/test ID overlap.

**Note on `test.csv`:** it has no `is_claim` column, so it cannot be used to evaluate models. All evaluation uses an **80/20 stratified split of `train.csv`** (46,873 train / 11,719 validation, `random_state=42`), which preserves the 93.6/6.4 class ratio in both halves.

---

## Analysis pipeline

```
Load & inspect → EDA → Data cleaning → Feature engineering → Multicollinearity
      → Classification → Model comparison → Clustering → Business insights → Final report
```

| Step | What it does | Where |
|---|---|---|
| 1. Load & inspect | Structure, missing values, IDs, target imbalance, dtypes, cardinality, stratified split | Notebook (sections 1.1 to 1.11) |
| 2. Feature engineering | Parse `max_power` / `max_torque` into numeric fields; Yes/No to 1/0; build `safety_score`, `power_to_weight_ratio`, `vehicle_age_category`; one-hot encode with rare-category grouping | Deck, slide 4 |
| 3. Multicollinearity | The book contains only 11 distinct vehicle models, so raw specs correlate at r = 0.90 to 0.96; features trimmed from 90 to 31 and VIF checked | Deck, slide 4 |
| 4. Imbalance handling | No adjustment vs class weighting vs SMOTE (fit on training data only) | Deck, slide 4 |
| 5. Modelling | Logistic Regression, Decision Tree, Random Forest | Deck, slides 5 to 7 |
| 6. Clustering | K-Means (K=4) risk segments | Deck, slide 8 |
| 7. Business insights | Underwriting and pricing recommendations | Deck, slides 9 to 10 |

The notebook currently covers Step 1. Later steps are summarised in the presentation deck.

---

## Key results

| Model | Balancing | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | None | 0.000 | 0.000 | 0.000 | 0.611 |
| Logistic Regression | Class weight | 0.085 | 0.593 | 0.148 | 0.611 |
| Decision Tree | None | 0.082 | 0.096 | 0.088 | 0.511 |
| Random Forest | None | 0.077 | 0.007 | 0.012 | 0.569 |
| **Random Forest** | **SMOTE** | **0.097** | **0.532** | **0.165** | **0.644** |

**Main findings**

- `policy_tenure` is the strongest driver in both the logistic regression and the Random Forest (about 46% of RF importance).
- `age_of_policyholder` and `age_of_car` together account for about 40% of RF importance.
- Area clusters C2 and C3 remain significant after controlling for vehicle and policyholder profile.
- Vehicle spec composites (power-to-weight, safety score, NCAP rating) matter far less.
- Four risk segments range from 4.64% (Newest & Safest) to 6.73% (Petrol Manual, Younger) claim rate.
- Predictive power is real but modest (ROC-AUC 0.644); richer behavioural data such as telematics is the recommended next investment.

**Recommended use:** flag policies for underwriting review with Random Forest + SMOTE, not as an automatic decline.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` on `data/test.csv` or `/content/train.csv` | Update the paths in the first code cell (see Quick start, step 2) |
| Strings not detected as `object` dtype in newer pandas (3.x) | Use `select_dtypes(include=['object', 'string'])` or an explicit column list |
| Different numbers from the deck | Keep `random_state=42` and the 80/20 stratified split exactly as in section 1.10 |
