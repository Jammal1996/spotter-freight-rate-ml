# spotter-freight-rate-ml

# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

---

# My solution

**Candidate:** Ranad Aljammal

**Loom walkthrough:** [PASTE LOOM LINK HERE]

**Report:** `Freight_Rate_Report_Ranad_Aljammal.pdf`

## Summary

The task is to predict the price (`posted_rate`) of 12,000 freight loads for November and December 2025,
using 48,000 loads from January to October 2025 with known prices.

| Model | Test used | Average error (MAPE) |
|---|---|---|
| Simple rule (typical $/mile by truck type x distance) | Sep-Oct holdout | 10.49% |
| First model: LightGBM | Sep-Oct holdout | 6.36% |
| First model: LightGBM | 5-fold expanding window | 5.52% |
| **Final model: linear trend + LightGBM** | **5-fold expanding window** | **4.11%** (1.78% on normal rows) |

Dollar error (MAE) on the same five-fold test: 130.1 for the first model, **97.4** for the final model.

## Key findings

- The validation file covers **later months** (Nov-Dec) than the training file (Jan-Oct), so this is a forecasting task and the data must be split by date, never randomly.
- Price depends mainly on distance, truck type, and weight. Longer trips cost less per mile. Reefer is the most expensive per mile, Dry Van the cheapest.
- Prices **rise about 0.7% every 30 days** through the year. Tree models cannot continue a trend into unseen months, so the final model includes a time trend.
- `quote_signal` is a noisy signal, but its daily average is informative.

## Data-quality issues and how they were handled

| Issue | What was found | What was done |
|---|---|---|
| Corrupted prices | 677 training rows (1.4%) are about 3 to 5 times too high or too low compared with similar loads (337 low, 340 high), in every month | Excluded from **training only**. They are still counted when errors are measured |
| Negative weights | 292 rows in train, 145 in validation | Treated as sign errors: `abs()` |
| Missing values | Weight: 300 train / 165 validation. Market index: 374 / 249 | Weight: train median. Market index: same-day median. Flag columns added |
| New cities | 8 cities appear only in validation (1,447 loads) | Model uses coordinates and distance, not city names |
| Duplicates | None found | None needed |

No validation rows are ever dropped: all 12,000 loads receive a prediction.

## Train / validation split

The real task predicts the future, so the data is split **by date**:

1. **Holdout test:** learn from Jan-Aug (38,477 loads), predict Sep-Oct (9,523 loads).
2. **Expanding-window cross-validation (5 folds):** for each of June, July, August, September and October, learn from all earlier months and predict that month.

A random split would let the model learn from the same weeks it is tested on and give overly optimistic scores.
Cleaning rules (for example the typical weight) are learned from training data only. The final model is trained on all labeled data (Jan-Oct).

| Month predicted | Learned from | Loads | MAE ($) | MAPE |
|---|---|---|---|---|
| June | Jan-May | 24,023 | 112.6 | 4.16% |
| July | Jan-Jun | 28,806 | 97.5 | 4.00% |
| August | Jan-Jul | 33,718 | 86.8 | 4.01% |
| September | Jan-Aug | 38,477 | 93.0 | 3.78% |
| October | Jan-Sep | 43,147 | 97.3 | 4.59% |
| **Average** | | | **97.4** | **4.11%** |

Errors are measured on all rows, including the corrupted ones. On normal rows only, the average error is 1.78%.

## Model

A two-part model, trained on the log of the price:

1. **Linear regression:** distance, truck type, weight, a time trend, and daily market signals. It carries the smooth patterns and the upward trend into November and December.
2. **LightGBM on the residuals:** learns what the linear part misses, such as route differences (from coordinates) and small interactions.

The final prediction is the two parts added together, converted back to dollars. Random seeds are fixed (`SEED = 42`), so results are reproducible.

## Repository structure

```
.
|-- README.md
|-- requirements.txt
|-- score.py                         # provided scorer (unmodified)
|-- solution.py                      # full pipeline (this is what you run)
|-- Freight_Rate_Report_Ranad_Aljammal.pdf
|-- validation_predictions.csv       # submission file: load_id,predicted_rate
|-- december_chart_predictions.csv   # filled December chart inputs (fed to score.py)
|-- data/
|   |-- train_test.csv
|   |-- validation.csv
|   |-- validation_predictions_template.csv
|   `-- december_chart_inputs.csv
|-- notebooks/
|   `-- 01_freight_rate_model.ipynb  # step-by-step exploration (Google Colab)
|-- scorer_results/
|   `-- candidate_december.png       # chart created by score.py
`-- reports/                         # figures and CV tables created by solution.py
```

## How to run

Requires Python 3.9 or newer.

```bash
python -m pip install -r requirements.txt
python solution.py
```

This takes roughly 30 seconds to a minute and does everything:

1. Loads the data and runs the quality checks
2. Runs the exploratory analysis and saves figures to `reports/`
3. Cleans the data and builds features
4. Runs the time-based validation (baseline, first LightGBM, 5-fold CV, error analysis)
5. Detects the corrupted labels and the time trend
6. Trains the final model and runs its 5-fold CV
7. Writes `validation_predictions.csv` and `december_chart_predictions.csv`
8. Runs `score.py` to validate both files and draw `scorer_results/candidate_december.png`

Options:

```bash
python solution.py --data-dir path/to/data     # data folder (default: data)
python solution.py --out-dir path/to/output    # where outputs are written (default: .)
python solution.py --skip-exploration          # faster: skip EDA and the first-model comparison
python solution.py --score-script path/to/score.py
```

To run the scorer by hand on the generated files:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_chart_predictions.csv
```

Note: `data/december_chart_inputs.csv` is the empty template from Spotter. The file passed to the scorer is
`december_chart_predictions.csv`, the same file with `predicted_rate` filled in.

### Running in Google Colab

```python
!git clone https://github.com/<your-username>/spotter-freight-rate-ml.git
%cd spotter-freight-rate-ml
!python -m pip install -q -r requirements.txt
!python solution.py
```

The notebook in `notebooks/` was written for Colab and sets a `DATA_DIR` path in its first cell (Google Drive).
Change it to your data folder if you run the notebook yourself.

## Limitations

- If the hidden November and December prices contain the same ~1.4% corrupted values, no model can predict them, and the real error will be higher than the 1.78% measured on normal rows.
- The upward price trend is assumed to continue. The predicted median price per mile for November ($2.16) and December ($2.18) is in line with September and October (about $2.16), but this cannot be tested without the real prices.
- `market_index` is lower in the validation file (average 0.93) than in training (1.08).
- The official metric is calculated by Spotter, so both dollar error (MAE) and percent error (MAPE) are reported.
- Other algorithms (for example XGBoost) were not compared. The main gains came from excluding corrupted labels and adding a time trend.
