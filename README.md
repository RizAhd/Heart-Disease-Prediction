# Heart Disease Prediction Project

This project implements a complete end-to-end heart disease prediction system using:
- Logistic Regression
- Decision Tree
- Random Forest
- Artificial Neural Network (TensorFlow/Keras)

It covers data loading, preprocessing, EDA, feature engineering, model training, hyperparameter tuning, evaluation, comparison, and model persistence.

## 1. Project Structure

```text
Haearty/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── heart_disease_final_assignment.ipynb
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   └── (generated results)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   └── train_and_evaluate.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 2. Prerequisites

- Python 3.10 or higher
- pip
- Windows PowerShell (or any terminal)

## 3. Setup

From the project root folder:

```powershell
python -m venv heartdiseaseprediction
.\heartdiseaseprediction\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Dataset

Primary expected file:
- `data/Heart_Disease_Prediction.csv`

The loader is synced to your column format and target labels (`Presence` / `Absence`).

If this file is not available, the code automatically downloads the UCI Cleveland dataset and saves it as `data/Heart_Disease_Prediction.csv`.

## 5. Run Full Training and Evaluation

Run this command from the project root:

```powershell
python -m src.train_and_evaluate
```

This generates:
- Saved models in `models/`
- Evaluation figures in `reports/figures/`
- `reports/model_comparison_summary.csv`
- `reports/detailed_results.json`
- `reports/outlier_handling_summary.csv`
- `reports/correlation_filter_summary.json`
- `reports/feature_importance.csv`
- `reports/feature_selection_summary.json`
- `reports/ann_tuning_results.csv`

## 6. Execute the Notebook

Start Jupyter:

```powershell
jupyter notebook
```

Then open:
- `notebooks/heart_disease_final_assignment.ipynb`

Run cells in order from top to bottom to reproduce analysis, visualizations, and model comparison.

## 7. Outputs You Should See

After successful run, expected artifacts include:
- `models/logistic_regression.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/ann_model.keras`
- `models/ann_preprocessor.joblib`
- `reports/figures/cm_logistic_regression.png`
- `reports/figures/cm_decision_tree.png`
- `reports/figures/cm_random_forest.png`
- `reports/figures/cm_artificial_neural_network.png`
- `reports/figures/roc_curves_comparison.png`
- `reports/figures/ann_training_history.png`
- `reports/model_comparison_summary.csv`
- `reports/detailed_results.json`
- `reports/outlier_handling_summary.csv`
- `reports/correlation_filter_summary.json`
- `reports/feature_importance.csv`
- `reports/feature_selection_summary.json`
- `reports/ann_tuning_results.csv`

## 8. Troubleshooting

1. TensorFlow install issues on Windows:
   - Ensure Python version is supported by your TensorFlow version.
   - Upgrade pip before installation.

2. Notebook kernel mismatch:
   - Select the same virtual environment (`heartdiseaseprediction`) as the notebook kernel.

3. If execution is slow:
   - Reduce GridSearch parameter ranges in `src/train_and_evaluate.py`.

## 9. Assignment Mapping

This implementation directly addresses the final assignment objectives:
- Data collection and understanding
- Data preprocessing (missing values, duplicate handling, outlier capping)
- EDA
- Feature engineering (correlation filtering + feature importance based selection)
- ML + DL model development
- Evaluation and comparison
- Hyperparameter tuning (GridSearch/RandomizedSearch + ANN random search)
- Documentation and reporting
