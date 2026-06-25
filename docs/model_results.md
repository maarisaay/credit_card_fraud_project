# Model Results

## Model comparison

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Baseline Logistic Regression | 0.7189 | 0.9722 |
| Logistic Regression + Feature Engineering | 0.7370 | 0.9752 |
| AutoGluon medium_quality | 0.8823 | 0.9824 |
| AutoGluon best_quality | 0.8910 | 0.9801 |

## Final Kedro model

| Model | Best C | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression po Optuna | 8.2698 | 0.7479 | 0.7190 | 0.9731 |