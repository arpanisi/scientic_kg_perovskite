"""Tabular regression baselines for perovskite device-performance prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.build_datasets import DEFAULT_SPLIT_KEY, split_name_for_key, validate_split_ratios
from src.data.prepare_inputs import (
    ColumnStats,
    InputBuildConfig,
    PRIMARY_COLUMNS,
    SECONDARY_COLUMNS,
    activated_features,
    is_leakage_column,
    normalize_value,
)
from src.data.prepare_outputs import PHYSICAL_LIMITS, TARGET_COLUMNS


TARGET_FIELDS = tuple(TARGET_COLUMNS.keys())
TARGET_COLUMN_NAMES = tuple(TARGET_COLUMNS.values())
NULL_STRINGS = {"", "unknown", "nan", "none", "null", "n/a"}
LEAKAGE_TARGET_TOKENS = ("PCE", "Voc", "Jsc", "FF")
SUPPORTED_REGRESSION_MODELS = (
    "ridge",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
)
SUPPORTED_FEATURE_REPRESENTATIONS = (
    "core",
    "core_secondary",
    "core_rare",
    "hierarchical",
)


@dataclass(frozen=True)
class RegressionData:
    """Prepared tabular regression data."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    split_key: str


def load_perovskite_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def prepare_regression_data(
    dataframe: pd.DataFrame,
    representation: str = "core_rare",
    split_key: str = DEFAULT_SPLIT_KEY,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    random_state: int = 17,
) -> RegressionData:
    validate_split_ratios(train_ratio, validation_ratio)
    filtered = valid_target_frame(dataframe)
    feature_columns = regression_feature_columns(filtered, representation=representation)
    x = filtered[feature_columns].copy()
    y = filtered[list(TARGET_COLUMN_NAMES)].rename(columns={value: key for key, value in TARGET_COLUMNS.items()})

    split_source = split_key if split_key in filtered.columns else DEFAULT_SPLIT_KEY
    split_labels = filtered[split_source].map(normalize_value)
    valid_split_mask = split_labels.astype(bool)
    x = x.loc[valid_split_mask]
    y = y.loc[valid_split_mask]
    split_labels = split_labels.loc[valid_split_mask]

    train_mask = split_labels.map(
        lambda key: split_name_for_key(key, train_ratio, validation_ratio, random_state) == "train"
    )
    test_mask = split_labels.map(
        lambda key: split_name_for_key(key, train_ratio, validation_ratio, random_state) == "test"
    )

    return RegressionData(
        x_train=x.loc[train_mask],
        x_test=x.loc[test_mask],
        y_train=y.loc[train_mask],
        y_test=y.loc[test_mask],
        split_key=split_source,
    )


def regression_feature_columns(dataframe: pd.DataFrame, representation: str = "core_rare") -> list[str]:
    if representation not in SUPPORTED_FEATURE_REPRESENTATIONS:
        expected = ", ".join(SUPPORTED_FEATURE_REPRESENTATIONS)
        raise ValueError(f"Unknown feature representation. Expected one of: {expected}.")

    columns = []
    if representation in {"core", "core_rare", "hierarchical"}:
        columns.extend(PRIMARY_COLUMNS)
    if representation in {"core_secondary", "hierarchical"}:
        columns.extend(PRIMARY_COLUMNS)
        columns.extend(SECONDARY_COLUMNS)
    if representation == "core_secondary":
        return existing_safe_columns(dataframe, columns)

    if representation in {"core_rare", "hierarchical"}:
        stats = build_column_stats(dataframe)
        config = InputBuildConfig(representation=representation)
        rare_columns = set()
        records = dataframe.astype(object).to_dict(orient="records")
        for row in records:
            for column, _value in activated_features(row, stats, config):
                rare_columns.add(column)
        columns.extend(sorted(rare_columns))

    return existing_safe_columns(dataframe, columns)


def existing_safe_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    seen = set()
    selected = []
    target_columns = set(TARGET_COLUMN_NAMES)
    for column in columns:
        if column in seen:
            continue
        if column not in dataframe.columns:
            continue
        if column in target_columns or is_target_leakage_column(column):
            continue
        selected.append(column)
        seen.add(column)
    return selected


def is_target_leakage_column(column: str) -> bool:
    if column in TARGET_COLUMN_NAMES:
        return True
    if column.startswith(("Ref_", "EQE_", "Stabilised_performance_", "Stability_", "Outdoor_")):
        return True
    if column.startswith("JV_") and any(token in column for token in LEAKAGE_TARGET_TOKENS):
        return True
    return False


def build_column_stats(dataframe: pd.DataFrame) -> ColumnStats:
    nonempty_counts = {}
    value_counts = {}
    for column in dataframe.columns:
        counts = {}
        nonempty = 0
        for raw_value in dataframe[column].tolist():
            value = normalize_value(raw_value)
            if not value:
                continue
            nonempty += 1
            counts[value] = counts.get(value, 0) + 1
        nonempty_counts[column] = nonempty
        value_counts[column] = counts
    return ColumnStats(
        row_count=len(dataframe),
        nonempty_counts=nonempty_counts,
        value_counts=value_counts,
    )


def valid_target_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    for column in TARGET_COLUMN_NAMES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(TARGET_COLUMN_NAMES))
    for field, column in TARGET_COLUMNS.items():
        low, high = PHYSICAL_LIMITS[field]
        frame = frame[(frame[column] >= low) & (frame[column] <= high)]
    return frame


def build_preprocessor(dataframe: pd.DataFrame, model_name: str = "ridge") -> ColumnTransformer:
    numeric_columns = []
    categorical_columns = []
    for column in dataframe.columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        numeric_ratio = numeric.notna().mean()
        if numeric_ratio >= 0.9:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    numeric_pipeline = Pipeline(
        steps=[
            ("coerce", FunctionTransformer(coerce_numeric_frame, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    if model_name == "ridge":
        categorical_pipeline = Pipeline(
            steps=[
                ("coerce", FunctionTransformer(coerce_categorical_frame, validate=False)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
            ]
        )
    else:
        categorical_pipeline = Pipeline(
            steps=[
                ("coerce", FunctionTransformer(coerce_categorical_frame, validate=False)),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def coerce_numeric_frame(values: Any) -> pd.DataFrame:
    return pd.DataFrame(values).apply(pd.to_numeric, errors="coerce")


def coerce_categorical_frame(values: Any) -> pd.DataFrame:
    frame = pd.DataFrame(values)
    return frame.where(pd.notna(frame), "missing").astype(str)


def build_regression_model(name: str, x_train: pd.DataFrame, random_state: int = 17) -> Pipeline:
    preprocessor = build_preprocessor(x_train, model_name=name)
    estimator = regression_estimator(name, random_state=random_state)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def regression_estimator(name: str, random_state: int = 17):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=24,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "gradient_boosting":
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                random_state=random_state,
            )
        )
    if name == "hist_gradient_boosting":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=random_state,
            )
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is required for --model xgboost. Install it with `pip install xgboost`."
            ) from exc
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    expected = ", ".join(SUPPORTED_REGRESSION_MODELS)
    raise ValueError(f"Unknown regression model. Expected one of: {expected}.")


def evaluate_regression_predictions(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict[str, Any]:
    predictions = pd.DataFrame(y_pred, columns=list(TARGET_FIELDS), index=y_true.index)
    metrics = {}
    for field in TARGET_FIELDS:
        truth = y_true[field].to_numpy(dtype=float)
        pred = predictions[field].to_numpy(dtype=float)
        mse = mean_squared_error(truth, pred)
        metrics[field] = {
            "mae": float(mean_absolute_error(truth, pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(truth, pred)),
        }
    return metrics


def run_regression_baseline(
    csv_path: str,
    model_name: str = "ridge",
    representation: str = "core_rare",
    split_key: str = DEFAULT_SPLIT_KEY,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    random_state: int = 17,
) -> dict[str, Any]:
    dataframe = load_perovskite_dataframe(csv_path)
    data = prepare_regression_data(
        dataframe,
        representation=representation,
        split_key=split_key,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )
    model = build_regression_model(model_name, data.x_train, random_state=random_state)
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    return {
        "model": model_name,
        "representation": representation,
        "split_key": data.split_key,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "feature_count": int(data.x_train.shape[1]),
        "feature_columns": list(data.x_train.columns),
        "rows_total": int(len(dataframe)),
        "rows_used": int(len(data.x_train) + len(data.x_test)),
        "train_rows": int(len(data.x_train)),
        "test_rows": int(len(data.x_test)),
        "metrics": evaluate_regression_predictions(data.y_test, predictions),
    }
