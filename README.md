# Heart Disease Prediction using Machine Learning & Deep Learning

## Overview

This project implements a **production-ready, end-to-end heart disease prediction system** using supervised machine learning and deep learning techniques. It demonstrates a complete data science workflow: from raw data ingestion through model training, hyperparameter optimization, evaluation, and comparison.

**Problem Statement:** Predict the presence or absence of heart disease in patients based on clinical attributes using multiple ML/DL approaches.

**Dataset:** UCI Cleveland Heart Disease dataset (270 records, 14 clinical features, binary classification)

---

## Quick Start (3 Steps)

```powershell
# 1. Create and activate virtual environment
python -m venv heartdiseaseprediction
.\heartdiseaseprediction\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training pipeline
python -m src.train_and_evaluate
```

Done! Check `reports/` and `models/` folders for results.

---

## Detailed Setup Instructions

### Prerequisites
- **Python:** 3.10 or higher
- **OS:** Windows, macOS, or Linux
- **Memory:** Minimum 4GB RAM
- **Disk Space:** ~500MB for dataset + models + outputs
- **Tools:** PowerShell (Windows) or any terminal

### Step 1: Create Virtual Environment

```powershell
cd c:\Users\rizla\Desktop\Haearty
python -m venv heartdiseaseprediction
```

### Step 2: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\heartdiseaseprediction\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\heartdiseaseprediction\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source heartdiseaseprediction/bin/activate
```

### Step 3: Upgrade pip and Install Dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## How to Run the Project

### Option A: Run Full Training Pipeline (Automated)

Trains all 4 models (Logistic Regression, Decision Tree, Random Forest, ANN) with hyperparameter tuning:

```powershell
python -m src.train_and_evaluate
```

**Expected runtime:** ~2-5 minutes  
**Output location:** `reports/` and `models/` folders

### Option B: Run Interactive Jupyter Notebook

Provides step-by-step walkthrough with visualizations:

```powershell
jupyter notebook
```

Then open: `notebooks/heart_disease_final_assignment.ipynb`

**How to use:**
1. Cell 1: Import libraries
2. Cell 2: Load dataset
3. Cells 3-4: Exploratory Data Analysis (EDA)
4. Cells 5-6: Data preprocessing & feature engineering
5. Cell 7: Train all ML models
6. Cell 8: Train ANN with tuning
7. Cell 9: Compare models and visualize results

---

## Project Structure

```
Haearty/
├── data/                                 # Dataset folder
│   └── Heart_Disease_Prediction.csv     # UCI dataset (auto-downloaded if missing)
│
├── models/                               # Trained model artifacts
│   ├── logistic_regression.joblib       # LR model
│   ├── decision_tree.joblib             # DT model
│   ├── random_forest.joblib             # RF model
│   ├── ann_model.keras                  # ANN model
│   └── ann_preprocessor.joblib          # Feature preprocessor
│
├── notebooks/
│   └── heart_disease_final_assignment.ipynb  # Complete interactive workflow
│
├── reports/                              # Analysis & results
│   ├── figures/                         # Generated plots
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_decision_tree.png
│   │   ├── cm_random_forest.png
│   │   ├── cm_artificial_neural_network.png
│   │   ├── roc_curves_comparison.png
│   │   └── ann_training_history.png
│   │
│   ├── model_comparison_summary.csv     # Final metrics table
│   ├── detailed_results.json            # Classification reports
│   ├── ann_tuning_results.csv          # ANN hyperparameter search results
│   ├── feature_importance.csv          # Feature rankings
│   ├── feature_selection_summary.json  # Selected features
│   ├── correlation_filter_summary.json # Correlation analysis
│   └── outlier_handling_summary.csv    # Outlier statistics
│
├── src/                                  # Source code
│   ├── __init__.py
│   ├── config.py                        # Central configuration
│   ├── data_loader.py                   # Dataset loading & preprocessing
│   └── train_and_evaluate.py            # Model training & evaluation
│
├── requirements.txt                      # Python dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

---

## Module Descriptions

### `src/config.py`
Central configuration file containing:
- Dataset file paths and column names
- Feature group definitions (numerical & categorical)
- Random state for reproducibility
- Model parameter grids

### `src/data_loader.py`
Handles dataset operations:
- Download UCI Cleveland dataset if local file missing
- Normalize column names and data types
- Handle target label mapping (Presence → 1, Absence → 0)
- Preprocess: duplicates, missing values, scaling, encoding

### `src/train_and_evaluate.py`
Main ML/DL pipeline:
- **Outlier handling:** IQR-based capping on numerical features
- **Feature engineering:** Correlation filtering + importance-based selection
- **ML models:** Logistic Regression (GridSearchCV), Decision Tree (GridSearchCV), Random Forest (RandomizedSearchCV)
- **DL model:** 4-layer ANN with parameterized architecture and random hyperparameter search
- **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices
- **Persistence:** Save all models for inference

---

## Expected Results Summary

After running the pipeline, you should see:

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 85.2% | 80.8% | 87.5% | 84.0% | **0.9014** ✓ |
| Random Forest | 83.3% | 80.0% | 83.3% | 81.6% | 0.8986 |
| Artificial Neural Network | 79.6% | 74.1% | 83.3% | 78.4% | 0.8944 |
| Decision Tree | 79.6% | 74.1% | 83.3% | 78.4% | 0.8840 |

**Best Performing Model:** Logistic Regression (ROC-AUC: 0.9014)

### Generated Output Files

**CSV Reports:**
- `model_comparison_summary.csv` – Metrics for all 4 models
- `ann_tuning_results.csv` – 8 ANN hyperparameter configurations tested
- `feature_importance.csv` – Feature rankings from Random Forest
- `outlier_handling_summary.csv` – Outliers detected & capped per feature

**JSON Reports:**
- `detailed_results.json` – Classification reports, confusion matrices
- `feature_selection_summary.json` – Top 10 selected features
- `correlation_filter_summary.json` – Feature correlation analysis

**Visualizations:**
- `roc_curves_comparison.png` – ROC curves for all 4 models
- `cm_*.png` – Confusion matrices for each model
- `ann_training_history.png` – ANN loss & accuracy curves

---

## Output Files Explained

| File | Content | Use Case |
|------|---------|----------|
| `model_comparison_summary.csv` | Accuracy, Precision, Recall, F1, ROC-AUC for each model | Selecting best model, performance report |
| `detailed_results.json` | Precision/recall/F1 per class, confusion matrices | Detailed analysis, class-specific metrics |
| `ann_tuning_results.csv` | 8 hyperparameter sets tested with validation scores | Understanding ANN tuning search space |
| `feature_importance.csv` | All features ranked by Random Forest importance | Feature selection rationale |
| `feature_selection_summary.json` | Top 10 selected features used in final models | Understanding model inputs |
| `outlier_handling_summary.csv` | Count & statistics of outliers capped per feature | Data quality assessment |
| `roc_curves_comparison.png` | Overlaid ROC curves for all 4 models | Visual model comparison |
| `cm_*.png` | Confusion matrix heatmaps | Classification pattern analysis |

---

## Data Preprocessing Pipeline

The project applies the following preprocessing steps:

1. **Data Loading:** Load UCI dataset with automatic column standardization
2. **Duplicate Removal:** Drop identical rows
3. **Missing Value Handling:** Use median imputation for numerical features
4. **Outlier Detection & Capping:** IQR method (cap extreme values)
5. **Categorical Encoding:** One-Hot Encoding for categorical features
6. **Feature Scaling:** StandardScaler normalization
7. **Feature Engineering:**
   - Correlation filtering (remove features with >0.90 correlation)
   - Random Forest importance-based selection (top 10 features)
8. **Train/Test Split:** 80/20 split with stratification

---

## Model Details

### Supervised Learning Models

**1. Logistic Regression (Best Model)**
- Hyperparameter tuning: C, penalty, solver
- Best parameters: Tuned via GridSearchCV
- Reason for performance: Simple, effective for binary classification

**2. Decision Tree**
- Hyperparameter tuning: max_depth, min_samples_split, min_samples_leaf
- Tuned via GridSearchCV
- Interpretable model, prone to overfitting on small datasets

**3. Random Forest**
- Hyperparameter tuning: n_estimators, max_depth, min_samples_split
- Tuned via RandomizedSearchCV (large parameter space)
- Ensemble method, provides feature importance rankings

### Deep Learning Model

**Artificial Neural Network (ANN)**
- Architecture: 4 fully connected layers with dropout regularization
- Layer sizes: Input → 96 → 16 → 8 → 1 (binary classification)
- Activation: ReLU (hidden), Sigmoid (output)
- Regularization: Dropout (tuned: 0.2-0.4)
- Optimizer: Adam (tuned learning rates: 0.001-0.01)
- Loss: Binary crossentropy
- Metrics: Accuracy, ROC-AUC
- Hyperparameter tuning: 8-iteration random search with validation split
- Early stopping: Training halted if validation loss doesn't improve
- Best validation ROC-AUC: 0.98125

---

## Troubleshooting

### Issue: Virtual Environment Not Activating
**Solution:**
- Check your Python path: `python --version`
- Ensure PowerShell execution policy allows scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Try Command Prompt instead: `.\heartdiseaseprediction\Scripts\activate.bat`

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:**
- Verify environment is activated (prompt shows `(heartdiseaseprediction)`)
- Reinstall dependencies: `pip install -r requirements.txt`
- For TensorFlow on Apple Silicon: `pip install tensorflow-macos`

### Issue: "FileNotFoundError: data/Heart_Disease_Prediction.csv"
**Solution:**
- Script auto-downloads if file missing, ensure internet connection
- Manually download from UCI: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- Place in `data/` folder as `Heart_Disease_Prediction.csv`

### Issue: Jupyter Notebook Kernel Issues
**Solution:**
- Remove old kernel: `jupyter kernelspec list` then `jupyter kernelspec remove heartdiseaseprediction`
- Reinstall: `pip install ipykernel`
- Register kernel: `python -m ipykernel install --user --name heartdiseaseprediction --display-name "Heart Disease (3.13)"`

### Issue: Script Taking Too Long (>10 minutes)
**Solution:**
- Reduce hyperparameter grid in `src/train_and_evaluate.py`
- Reduce `n_iter` for RandomizedSearchCV (default: 20)
- Close other applications to free memory

---

## Dataset Information

**Source:** [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

**Dataset Statistics:**
- **Records:** 270 patients
- **Features:** 14 (13 input features + 1 target)
- **Target:** Binary classification (Heart Disease: Present/Absent)
- **Class Distribution:** 150 Present (55.6%), 120 Absent (44.4%)
- **Missing Values:** Handled via imputation
- **Outliers:** 15 outliers capped via IQR method

**Features (13 input):**
- **Numerical (5):** age, trestbps (resting BP), chol (cholesterol), thalach (max HR), oldpeak (ST depression)
- **Categorical (8):** sex, cp (chest pain type), fbs (fasting blood sugar), restecg, exang (exercise induced angina), slope, ca (coronary artery count), thal (thalassemia)

---

## Assignment Mapping

This project fulfills all final assignment requirements:

✓ **Problem Definition:** Heart disease prediction from clinical features  
✓ **Supervised Learning:** 3 ML models (LR, DT, RF) with hyperparameter tuning  
✓ **Deep Learning:** ANN with architecture & hyperparameter optimization  
✓ **Data Preprocessing:** Outlier handling, scaling, encoding, imputation  
✓ **EDA & Visualization:** Distributions, correlations, ROC curves  
✓ **Feature Engineering:** Correlation filtering + importance selection  
✓ **Model Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC  
✓ **Model Comparison:** Summary table with all 4 models  
✓ **Reproducibility:** Modular code, central config, named environment  
✓ **Documentation:** Comprehensive README, inline comments, Jupyter notebook  

---

## Requirements & Dependencies

See `requirements.txt` for full list. Key packages:

```
pandas==3.0.1              # Data manipulation
numpy==2.4.3              # Numerical computing
scikit-learn==1.6.1       # ML models & evaluation
tensorflow==2.17.0        # Deep learning
keras==3.4.1              # ANN framework
matplotlib==3.10.8        # Plotting
seaborn==0.13.2           # Statistical visualization
joblib==1.4.2             # Model serialization
jupyter==1.0.0            # Notebook environment
```

---

## References

1. **UCI Heart Disease Dataset:**  
   Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease Data Set. UCI Machine Learning Repository.  
   https://archive.ics.uci.edu/ml/datasets/Heart+Disease

2. **Scikit-learn Documentation:**  
   https://scikit-learn.org/stable/

3. **TensorFlow & Keras Documentation:**  
   https://www.tensorflow.org/

4. **Hyperparameter Tuning:**  
   Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13, 281-305.

5. **ROC-AUC and Model Evaluation:**  
   Fawcett, T. (2006). An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861-874.

---

## Author & License

**Project:** Heart Disease Prediction - ML/DL Assignment  
**Institution:** ICBS International - Diploma in Data Science & Machine Learning  
**Batch:** 04  
**Date:** March 2026

---

## Support & Questions

For issues, questions, or improvements:
1. Check the **Troubleshooting** section above
2. Review source code comments in `src/` folder
3. Refer to Jupyter notebook cells for step-by-step walkthrough
