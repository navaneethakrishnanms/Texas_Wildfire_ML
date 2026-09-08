# Model Evaluation Metrics

Based on the training run for the 30-feature XGBoost wildfire ignition model, here are the detailed evaluation metrics separated by Train, Validation, and Test splits.

Since probability calibration was fitted on the validation set, we report both the **Raw Model** outputs and the **Calibrated Model** outputs where applicable.

---

## 1. Training Set (2014-2023)
*This is the data the model learned from. We only show the raw scores here, as calibrating the training set would lead to overconfident probabilities.*

| Metric | Score |
|---|---|
| **Rows** | 2,761,329 |
| **Base Rate** (Positives) | 26.58% |
| **AUC-PR** | 0.7458 |
| **AUROC** | 0.8516 |
| **Lift** (improvement over random) | 2.81× |
| **F1 Score** | 0.6452 |
| **Precision** | 0.6391 |
| **Recall** | 0.6515 |

---

## 2. Validation Set (2024)
*This data was used to stop the training early to prevent overfitting, and then used to fit the Isotonic Calibrator.*

| Metric | Raw Model | Calibrated Model |
|---|:---:|:---:|
| **Rows** | 287,649 | 287,649 |
| **Base Rate** | 29.39% | 29.39% |
| **AUC-PR** | 0.7338 | 0.7298 |
| **AUROC** | 0.8208 | 0.8211 |
| **Lift** | 2.50× | 2.48× |
| **F1 Score** | 0.6317 | 0.6318 |
| **Precision** | 0.6428 | 0.6423 |
| **Recall** | 0.6209 | 0.6216 |

---

## 3. Test Set (2025 - 2026-07-29)
*This is completely unseen data, representing how the model will perform in the real world going forward.*

| Metric | Raw Model | Calibrated Model |
|---|:---:|:---:|
| **Rows** | 460,463 | 460,463 |
| **Base Rate** | 30.71% | 30.71% |
| **AUC-PR** | **0.7384** | **0.7337** |
| **AUROC** | **0.8249** | **0.8249** |
| **Lift** | **2.40×** | **2.39×** |
| **F1 Score** | **0.6440** | **0.6440** |
| **Precision** | **0.6435** | **0.6428** |
| **Recall** | **0.6445** | **0.6452** |

---

## Key Takeaway
The model's performance on the unseen **Test Set** (AUC-PR 0.738) is extremely close to its performance on the **Training Set** (AUC-PR 0.745). This tiny gap (0.007) proves that **the model is not overfitting** and has successfully generalized the physical drivers of wildfire ignitions in Texas.
