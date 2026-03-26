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
from sklearn.model_selection import GridSearchCV, ParameterSampler, RandomizedSearchCV, train_test_split
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


def infer_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_features = [c for c in NUMERICAL_FEATURES if c in X.columns]
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    return num_features, cat_features


def create_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
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
            ("num", numeric_pipe, num_features),
            ("cat", categorical_pipe, cat_features),
        ]
    )
    return preprocessor


def handle_outliers_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    summary_rows = []

    for col in NUMERICAL_FEATURES:
        if col not in cleaned.columns:
            continue

        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before_low = int((cleaned[col] < lower).sum())
        before_high = int((cleaned[col] > upper).sum())

        cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

        summary_rows.append(
            {
                "feature": col,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "low_outliers_capped": before_low,
                "high_outliers_capped": before_high,
                "total_capped": before_low + before_high,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return cleaned, summary_df


def apply_correlation_filter(df: pd.DataFrame, threshold: float = 0.90) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in df.columns and c != TARGET_COL]
    if not numeric_cols:
        return df, []

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]

    filtered = df.drop(columns=to_drop, errors="ignore")
    return filtered, to_drop


def select_top_features_by_importance(X_train: pd.DataFrame, y_train: pd.Series, top_k: int = 10) -> tuple[list[str], pd.DataFrame]:
    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)

    X_imp = X_train.copy()
    for col in X_imp.columns:
        if X_imp[col].isna().any():
            if col in NUMERICAL_FEATURES:
                X_imp[col] = X_imp[col].fillna(X_imp[col].median())
            else:
                X_imp[col] = X_imp[col].fillna(X_imp[col].mode(dropna=True).iloc[0])

    model.fit(X_imp, y_train)
    importances = pd.DataFrame(
        {
            "feature": X_imp.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    k = max(1, min(top_k, len(importances)))
    selected = importances.head(k)["feature"].tolist()
    return selected, importances


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
    num_features, cat_features = infer_feature_groups(X_train)
    preprocessor = create_preprocessor(num_features, cat_features)

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


def build_ann(input_dim: int, params: dict) -> Sequential:
    model = Sequential(
        [
            Dense(params["units_1"], activation="relu", input_shape=(input_dim,)),
            Dropout(params["dropout_1"]),
            Dense(params["units_2"], activation="relu"),
            Dropout(params["dropout_2"]),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_ann(X_train, X_test, y_train, y_test) -> tuple[dict, pd.DataFrame, tuple[np.ndarray, np.ndarray, float]]:
    num_features, cat_features = infer_feature_groups(X_train)
    preprocessor = create_preprocessor(num_features, cat_features)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    if hasattr(X_tr_proc, "toarray"):
        X_tr_proc = X_tr_proc.toarray()
        X_val_proc = X_val_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    param_space = {
        "units_1": [32, 64, 96],
        "units_2": [16, 32, 48],
        "dropout_1": [0.2, 0.3, 0.4],
        "dropout_2": [0.1, 0.2, 0.3],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [16, 32],
    }
    sampled_params = list(ParameterSampler(param_space, n_iter=8, random_state=RANDOM_STATE))

    best_auc = -1.0
    best_model = None
    best_params = None
    best_history_df = None
    tuning_rows = []

    for params in sampled_params:
        ann = build_ann(X_tr_proc.shape[1], params)
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        history = ann.fit(
            X_tr_proc,
            y_tr,
            validation_data=(X_val_proc, y_val),
            epochs=150,
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
            verbose=0,
        )

        val_score = ann.predict(X_val_proc, verbose=0).reshape(-1)
        val_auc = roc_auc_score(y_val, val_score)
        tuning_rows.append({**params, "val_roc_auc": float(val_auc), "epochs_ran": len(history.history["loss"])})

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = ann
            best_params = params
            best_history_df = pd.DataFrame(history.history)

    tuning_df = pd.DataFrame(tuning_rows).sort_values("val_roc_auc", ascending=False)
    tuning_df.to_csv(REPORTS_DIR / "ann_tuning_results.csv", index=False)

    y_score = best_model.predict(X_test_proc, verbose=0).reshape(-1)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "model": "Artificial Neural Network",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "best_params": best_params,
        "best_validation_roc_auc": float(best_auc),
    }

    save_confusion_matrix(y_test, y_pred, "Artificial Neural Network")

    fpr, tpr, _ = roc_curve(y_test, y_score)

    best_model.save(MODELS_DIR / "ann_model.keras")
    joblib.dump(preprocessor, MODELS_DIR / "ann_preprocessor.joblib")

    return metrics, best_history_df, (fpr, tpr, metrics["roc_auc"])


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

    df, outlier_summary = handle_outliers_iqr(df)
    if not outlier_summary.empty:
        outlier_summary.to_csv(REPORTS_DIR / "outlier_handling_summary.csv", index=False)

    df, dropped_by_corr = apply_correlation_filter(df, threshold=0.90)
    with open(REPORTS_DIR / "correlation_filter_summary.json", "w", encoding="utf-8") as fp:
        json.dump({"threshold": 0.90, "dropped_features": dropped_by_corr}, fp, indent=2)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    selected_features, feature_importance = select_top_features_by_importance(X_train, y_train, top_k=10)
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    with open(REPORTS_DIR / "feature_selection_summary.json", "w", encoding="utf-8") as fp:
        json.dump({"selected_top_features": selected_features}, fp, indent=2)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

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
            "best_params": ann_metrics["best_params"],
            "best_validation_roc_auc": ann_metrics["best_validation_roc_auc"],
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
