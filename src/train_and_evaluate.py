from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FIGURES_DIR,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)
from src.data_loader import load_dataset


sns.set_theme(style="whitegrid")
np.random.seed(RANDOM_STATE)


def ensure_output_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    return df


def create_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERICAL_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_score": y_score,
    }
    return metrics


def save_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


def save_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    filename: str = "roc_curves_comparison.png",
    title: str = "ROC Curves - Model Comparison",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, (fpr, tpr, auc_val) in curves.items():
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def train_ml_models(X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict, dict[str, tuple[np.ndarray, np.ndarray, float]]]:
    preprocessor = create_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }

    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"],
        },
        "Decision Tree": {
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    }

    metrics_rows = []
    roc_data = {}

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

        if name == "Random Forest":
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grids[name],
                n_iter=10,
                scoring="roc_auc",
                cv=CV_FOLDS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            search = GridSearchCV(pipe, param_grid=param_grids[name], scoring="roc_auc", cv=CV_FOLDS, n_jobs=-1)

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        model_metrics = evaluate_model(name, best_model, X_test, y_test)
        model_metrics["best_params"] = search.best_params_
        metrics_rows.append(model_metrics)

        y_pred = best_model.predict(X_test)
        save_confusion_matrix(y_test, y_pred, name)

        fpr, tpr, _ = roc_curve(y_test, model_metrics["y_score"])
        roc_data[name] = (fpr, tpr, model_metrics["roc_auc"])

        joblib.dump(best_model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    summary = pd.DataFrame(
        [
            {
                "model": r["model"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
                "roc_auc": r["roc_auc"],
            }
            for r in metrics_rows
        ]
    ).sort_values(by="roc_auc", ascending=False)

    details = {r["model"]: {"best_params": r["best_params"], "classification_report": r["classification_report"]} for r in metrics_rows}
    return summary, details, roc_data


def build_ann(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dropout(0.30),
            Dense(32, activation="relu"),
            Dropout(0.20),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_ann(X_train, X_test, y_train, y_test) -> tuple[dict, pd.DataFrame, tuple[np.ndarray, np.ndarray, float]]:
    preprocessor = create_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    ann = build_ann(X_train_proc.shape[1])
    early_stopping = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    history = ann.fit(
        X_train_proc,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=0,
    )

    y_score = ann.predict(X_test_proc, verbose=0).reshape(-1)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "model": "Artificial Neural Network",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    save_confusion_matrix(y_test, y_pred, "Artificial Neural Network")

    fpr, tpr, _ = roc_curve(y_test, y_score)

    ann.save(MODELS_DIR / "ann_model.keras")
    joblib.dump(preprocessor, MODELS_DIR / "ann_preprocessor.joblib")

    history_df = pd.DataFrame(history.history)
    return metrics, history_df, (fpr, tpr, metrics["roc_auc"])


def save_history_plot(history_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history_df["loss"], label="Train Loss")
    ax.plot(history_df["val_loss"], label="Validation Loss")
    ax.set_title("ANN Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Crossentropy Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ann_training_history.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()

    df = load_dataset()
    df = preprocess_input(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    ml_summary, ml_details, ml_roc_data = train_ml_models(X_train, X_test, y_train, y_test)

    ann_metrics, ann_history, ann_roc = train_ann(X_train, X_test, y_train, y_test)
    save_history_plot(ann_history)

    all_roc_data = dict(ml_roc_data)
    all_roc_data["Artificial Neural Network"] = ann_roc
    save_roc_curves(all_roc_data)

    full_summary = pd.concat([ml_summary, pd.DataFrame([ann_metrics]).drop(columns=["classification_report"])], ignore_index=True)
    full_summary = full_summary.sort_values(by="roc_auc", ascending=False)

    best_model_name = full_summary.iloc[0]["model"]

    full_summary.to_csv(REPORTS_DIR / "model_comparison_summary.csv", index=False)

    detailed_results = {
        "ml_model_details": ml_details,
        "ann_details": {
            "classification_report": ann_metrics["classification_report"],
        },
        "best_model": best_model_name,
    }

    with open(REPORTS_DIR / "detailed_results.json", "w", encoding="utf-8") as fp:
        json.dump(detailed_results, fp, indent=2)

    print("Training complete.")
    print(f"Best model: {best_model_name}")
    print("Artifacts saved in models/ and reports/.")


if __name__ == "__main__":
    main()
