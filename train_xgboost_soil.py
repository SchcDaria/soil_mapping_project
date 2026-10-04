from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


SOURCE_DEFAULT = str(Path(__file__).resolve().parent / "data/training/soil_relief_tlu.csv")
OUTPUT_DEFAULT = "soil_xgboost_outputs"
RANDOM_STATE_DEFAULT = 42

TARGET_COL = "Soil"
REQUIRED_NUMERIC_SOURCE_COLS = ["height_1", "slope_1", "aspect_1", "profile_1", "tangential_1"]
OPTIONAL_RELIEF_SOURCE_COLS = [
    "TPI_1",
    "TRI_1",
    "roughness_1",
    "flow_1",
    "twi_1",
    "dist_stream_1",
    "osm_water_1",
    "osm_wetland_1",
    "osm_waterbody_1",
    "osm_road_1",
    "tpi_3x3_1",
    "tpi_7x7_1",
    "tpi_15x15_1",
    "rough_3x3_1",
    "rough_7x7_1",
    "rough_15x15_1",
    "relief_std_3x3_1",
    "relief_std_7x7_1",
    "relief_std_15x15_1",
]
BASE_RELIEF_FEATURES = [
    "height_1",
    "slope_1",
    "aspect_sin",
    "aspect_cos",
    "profile_1",
    "tangential_1",
]
TAXATION_FEATURES = ["TLU"]
SOIL_AUX_FEATURES = ["GRAN", "MP"]
ID_COLS = ["fid", "WKT"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    numeric_features: list[str]
    categorical_features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean soil profile data and train three XGBoost classifiers for "
            "digital soil mapping."
        )
    )
    parser.add_argument("--input", default=SOURCE_DEFAULT, help="Path to soil_relief_tlu.csv.")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Directory for model artifacts.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Stratified test split size.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, encoding="utf-8-sig")


def find_tlu_column(columns: pd.Index) -> str:
    if "TLU" in columns:
        return "TLU"

    candidates = [col for col in columns if "TLU" in str(col).upper() or "ТЛУ" in str(col).upper()]
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely identify the TLU column. "
            f"Candidates found: {candidates}. Columns: {list(columns)}"
        )
    return candidates[0]


def clean_string_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})
    return cleaned


def get_relief_features(df: pd.DataFrame) -> list[str]:
    return BASE_RELIEF_FEATURES + [col for col in OPTIONAL_RELIEF_SOURCE_COLS if col in df.columns]


def build_model_specs(relief_features: list[str]) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="model_1_relief_tlu",
            description="Soil = f(relief, TLU)",
            numeric_features=relief_features,
            categorical_features=TAXATION_FEATURES,
        ),
        ModelSpec(
            name="model_2_relief_only",
            description="Soil = f(relief)",
            numeric_features=relief_features,
            categorical_features=[],
        ),
        ModelSpec(
            name="model_3_relief_tlu_gran_mp",
            description="Soil = f(relief, TLU, GRAN, MP)",
            numeric_features=relief_features,
            categorical_features=TAXATION_FEATURES + SOIL_AUX_FEATURES,
        ),
    ]


def add_coordinates_from_wkt(df: pd.DataFrame) -> pd.DataFrame:
    if "WKT" not in df.columns:
        return df

    coords = df["WKT"].astype("string").str.extract(
        r"MULTIPOINT\s*\(\(\s*(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s*\)\)",
        expand=True,
    )
    if coords.notna().any().any():
        df["x"] = pd.to_numeric(coords["x"], errors="coerce")
        df["y"] = pd.to_numeric(coords["y"], errors="coerce")
    return df


def clean_data(input_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(input_path)
    df = raw.copy()

    tlu_col = find_tlu_column(df.columns)
    if tlu_col != "TLU":
        df = df.rename(columns={tlu_col: "TLU"})

    for col in [TARGET_COL, "GRAN", "MP", "TLU"]:
        if col in df.columns:
            df[col] = clean_string_column(df[col])

    numeric_source_cols = REQUIRED_NUMERIC_SOURCE_COLS + [
        col for col in OPTIONAL_RELIEF_SOURCE_COLS if col in df.columns
    ]

    for col in REQUIRED_NUMERIC_SOURCE_COLS:
        if col not in df.columns:
            raise ValueError(f"Required numeric column is missing: {col}")

    for col in numeric_source_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_coordinates_from_wkt(df)
    before_rows = len(df)

    df = df.dropna(subset=[TARGET_COL]).copy()
    df = df.dropna(subset=REQUIRED_NUMERIC_SOURCE_COLS).copy()
    df["TLU"] = df["TLU"].fillna("__MISSING_TLU__")
    df["GRAN"] = df["GRAN"].fillna("__MISSING_GRAN__")
    df["MP"] = df["MP"].fillna("__MISSING_MP__")

    aspect_radians = np.deg2rad(df["aspect_1"] % 360.0)
    df["aspect_sin"] = np.sin(aspect_radians)
    df["aspect_cos"] = np.cos(aspect_radians)

    duplicate_rows = int(df.duplicated().sum())
    if "fid" in df.columns:
        duplicate_fids = int(df["fid"].duplicated().sum())
    else:
        duplicate_fids = None

    profile = {
        "input_path": str(input_path),
        "raw_rows": int(before_rows),
        "clean_rows": int(len(df)),
        "dropped_rows": int(before_rows - len(df)),
        "renamed_tlu_column_from": tlu_col,
        "duplicate_rows_after_cleaning": duplicate_rows,
        "duplicate_fids_after_cleaning": duplicate_fids,
        "missing_values_after_cleaning": df.isna().sum().to_dict(),
        "class_counts": df[TARGET_COL].value_counts().to_dict(),
        "tlu_counts": df["TLU"].value_counts().to_dict(),
        "gran_counts": df["GRAN"].value_counts().to_dict(),
        "mp_counts": df["MP"].value_counts().to_dict(),
        "numeric_source_columns": numeric_source_cols,
        "relief_features_used": get_relief_features(df),
        "numeric_describe": df[numeric_source_cols + ["aspect_sin", "aspect_cos"]]
        .describe()
        .to_dict(),
    }
    return df, profile


def make_pipeline(
    spec: ModelSpec,
    num_classes: int,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "numeric",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            spec.numeric_features,
        )
    ]

    if spec.categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                spec.categorical_features,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", classifier)])


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    return metrics, report_df, matrix_df


def save_feature_importance(pipeline: Pipeline, path: Path) -> None:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    write_csv(importance.reset_index(drop=True), path)


def train_models(
    df: pd.DataFrame,
    output_dir: Path,
    model_specs: list[ModelSpec],
    test_size: float,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL])

    train_index, test_index = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    y_train = y[df.index.get_indexer(train_index)]
    y_test = y[df.index.get_indexer(test_index)]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    split_cols = [col for col in ID_COLS + [TARGET_COL] if col in df.columns]
    write_csv(df.loc[train_index, split_cols], output_dir / "splits" / "train_rows.csv")
    write_csv(df.loc[test_index, split_cols], output_dir / "splits" / "test_rows.csv")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    summary_rows: list[dict[str, Any]] = []

    for spec in model_specs:
        model_dir = output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = spec.numeric_features + spec.categorical_features
        X_train = df.loc[train_index, feature_cols]
        X_test = df.loc[test_index, feature_cols]

        pipeline = make_pipeline(
            spec=spec,
            num_classes=len(label_encoder.classes_),
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        metrics, report_df, matrix_df = evaluate_predictions(y_test, y_pred, label_encoder.classes_)

        model_payload = {
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "model_spec": asdict(spec),
            "target": TARGET_COL,
            "feature_columns": feature_cols,
            "classes": label_encoder.classes_.tolist(),
        }
        joblib.dump(model_payload, model_dir / f"{spec.name}.joblib")
        write_json(
            model_dir / "metrics.json",
            {
                "model": asdict(spec),
                "metrics": metrics,
                "xgboost_params": pipeline.named_steps["model"].get_params(),
            },
        )
        write_csv(report_df, model_dir / "classification_report.csv")
        write_csv(matrix_df, model_dir / "confusion_matrix.csv")
        save_feature_importance(pipeline, model_dir / "feature_importance.csv")

        predictions = df.loc[test_index, [col for col in ID_COLS if col in df.columns]].copy()
        predictions["true_soil"] = label_encoder.inverse_transform(y_test)
        predictions["predicted_soil"] = label_encoder.inverse_transform(y_pred)
        predictions["predicted_probability"] = y_proba.max(axis=1)
        for class_index, class_name in enumerate(label_encoder.classes_):
            predictions[f"prob_{class_name}"] = y_proba[:, class_index]
        write_csv(predictions, model_dir / "test_predictions.csv")

        summary_rows.append(
            {
                "model": spec.name,
                "description": spec.description,
                "features": ", ".join(feature_cols),
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    write_csv(summary, output_dir / "summary_metrics.csv")
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, profile = clean_data(input_path)
    relief_features = get_relief_features(df)
    model_specs = build_model_specs(relief_features)
    write_csv(df, output_dir / "cleaned_soil_relief_tlu.csv")
    write_json(output_dir / "data_profile.json", profile)
    write_json(
        output_dir / "run_config.json",
        {
            "input": str(input_path),
            "output": str(output_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "relief_feature_note": (
                "aspect_1 is converted to aspect_sin and aspect_cos to preserve "
                "the circular nature of exposition/aspect. If TPI_1, TRI_1, "
                "and roughness_1 exist in the input CSV, they are also used as "
                "relief features."
            ),
            "relief_features_used": relief_features,
            "models": [asdict(spec) for spec in model_specs],
        },
    )

    summary = train_models(
        df=df,
        output_dir=output_dir,
        model_specs=model_specs,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )

    print("\nTraining complete. Summary metrics:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
