from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from train_xgboost_soil import (
    ID_COLS,
    RANDOM_STATE_DEFAULT,
    SOURCE_DEFAULT,
    TARGET_COL,
    build_model_specs,
    clean_data,
    evaluate_predictions,
    get_relief_features,
    make_pipeline,
    save_feature_importance,
    write_csv,
    write_json,
)


FULL_SOURCE_DEFAULT = str(Path(__file__).resolve().parent / "data/training/soil_relief_extended_full.csv")
OUTPUT_DEFAULT = "soil_xgboost_full_smote_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train three XGBoost soil classifiers with SMOTE/SMOTENC resampling."
    )
    parser.add_argument("--input", default=FULL_SOURCE_DEFAULT or SOURCE_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--sampling-strategy",
        default="not majority",
        help="SMOTE sampling_strategy. Default oversamples all classes except the majority.",
    )
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    return parser.parse_args()


def class_counts(y: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    counts = pd.Series(y).value_counts().sort_index()
    return {str(labels[int(index)]): int(value) for index, value in counts.items()}


def make_sampler(
    X_train: pd.DataFrame,
    categorical_features: list[str],
    y_train: np.ndarray,
    sampling_strategy: str,
    requested_k_neighbors: int,
    random_state: int,
) -> tuple[SMOTE | SMOTENC, dict[str, Any]]:
    min_class_count = int(pd.Series(y_train).value_counts().min())
    effective_k_neighbors = min(requested_k_neighbors, min_class_count - 1)
    if effective_k_neighbors < 1:
        raise ValueError(
            "SMOTE needs at least 2 training samples in every class. "
            f"Minimum class count in train split is {min_class_count}."
        )

    categorical_indices = [X_train.columns.get_loc(col) for col in categorical_features]
    if categorical_indices:
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=effective_k_neighbors,
        )
        sampler_name = "SMOTENC"
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=effective_k_neighbors,
        )
        sampler_name = "SMOTE"

    report = {
        "sampler": sampler_name,
        "sampling_strategy": sampling_strategy,
        "requested_k_neighbors": requested_k_neighbors,
        "effective_k_neighbors": effective_k_neighbors,
        "categorical_features": categorical_features,
        "categorical_indices": categorical_indices,
    }
    return sampler, report


def train_models_with_smote(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    sampling_strategy: str,
    smote_k_neighbors: int,
) -> pd.DataFrame:
    relief_features = get_relief_features(df)
    model_specs = build_model_specs(relief_features)

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

    split_cols = [col for col in ID_COLS + [TARGET_COL] if col in df.columns]
    write_csv(df.loc[train_index, split_cols], output_dir / "splits" / "train_rows.csv")
    write_csv(df.loc[test_index, split_cols], output_dir / "splits" / "test_rows.csv")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    summary_rows: list[dict[str, Any]] = []

    for spec in model_specs:
        model_dir = output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = spec.numeric_features + spec.categorical_features
        X_train = df.loc[train_index, feature_cols].copy()
        X_test = df.loc[test_index, feature_cols].copy()

        sampler, sampler_report = make_sampler(
            X_train=X_train,
            categorical_features=spec.categorical_features,
            y_train=y_train,
            sampling_strategy=sampling_strategy,
            requested_k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        pipeline = make_pipeline(
            spec=spec,
            num_classes=len(label_encoder.classes_),
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        pipeline.fit(X_resampled, y_resampled)

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
            "resampling": sampler_report,
        }
        joblib.dump(model_payload, model_dir / f"{spec.name}.joblib")
        write_json(
            model_dir / "metrics.json",
            {
                "model": asdict(spec),
                "metrics": metrics,
                "resampling": {
                    **sampler_report,
                    "train_class_counts_before": class_counts(y_train, label_encoder.classes_),
                    "train_class_counts_after": class_counts(y_resampled, label_encoder.classes_),
                },
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
                "resampling": sampler_report["sampler"],
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
            "experiment": "smote",
            "input": str(input_path),
            "output": str(output_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "sampling_strategy": args.sampling_strategy,
            "smote_k_neighbors": args.smote_k_neighbors,
            "relief_features_used": relief_features,
            "models": [asdict(spec) for spec in model_specs],
        },
    )

    summary = train_models_with_smote(
        df=df,
        output_dir=output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        sampling_strategy=args.sampling_strategy,
        smote_k_neighbors=args.smote_k_neighbors,
    )

    print("\nSMOTE training complete. Summary metrics:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
