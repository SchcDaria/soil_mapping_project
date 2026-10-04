from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from train_xgboost_soil import (
    ID_COLS,
    RANDOM_STATE_DEFAULT,
    SOIL_AUX_FEATURES,
    TAXATION_FEATURES,
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
OUTPUT_DEFAULT = "soil_xgboost_full_merge_rare_outputs"
MERGED_TARGET_COL = "Soil_merged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge rare soil classes into nearest larger classes using class vectors, "
            "then train three XGBoost classifiers."
        )
    )
    parser.add_argument("--input", default=FULL_SOURCE_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=30,
        help="Classes with fewer samples than this are merged into the nearest larger class.",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance used between class centroid vectors.",
    )
    return parser.parse_args()


def make_vector_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
    if categorical_features:
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
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_class_vectors(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    distance_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = numeric_features + categorical_features
    preprocessor = make_vector_preprocessor(numeric_features, categorical_features)
    matrix = preprocessor.fit_transform(df[feature_cols])
    feature_names = preprocessor.get_feature_names_out()

    centroid_rows: list[pd.Series] = []
    class_names: list[str] = []
    for class_name, index in df.groupby(TARGET_COL).groups.items():
        class_names.append(str(class_name))
        positions = df.index.get_indexer(index)
        centroid_rows.append(pd.Series(np.asarray(matrix[positions]).mean(axis=0), index=feature_names))

    centroids = pd.DataFrame(centroid_rows, index=class_names)
    distances = pd.DataFrame(
        pairwise_distances(centroids.to_numpy(), metric=distance_metric),
        index=centroids.index,
        columns=centroids.index,
    )

    nearest_rows: list[dict[str, Any]] = []
    for class_name in distances.index:
        ordered = distances.loc[class_name].drop(index=class_name).sort_values()
        for rank, (neighbor, distance) in enumerate(ordered.items(), start=1):
            nearest_rows.append(
                {
                    "class": class_name,
                    "neighbor_rank": rank,
                    "neighbor_class": neighbor,
                    "distance": float(distance),
                }
            )
    nearest = pd.DataFrame(nearest_rows)
    return centroids, distances, nearest


def build_merge_mapping(
    df: pd.DataFrame,
    distances: pd.DataFrame,
    min_class_count: int,
) -> pd.DataFrame:
    counts = df[TARGET_COL].value_counts()
    rare_classes = set(counts[counts < min_class_count].index.astype(str))
    major_classes = set(counts[counts >= min_class_count].index.astype(str))
    if rare_classes and not major_classes:
        raise ValueError(
            "All classes are rare under the selected threshold. "
            "Lower --min-class-count."
        )

    rows: list[dict[str, Any]] = []
    for class_name, count in counts.items():
        class_name = str(class_name)
        is_rare = class_name in rare_classes
        if is_rare:
            nearest_major = distances.loc[class_name, list(major_classes)].sort_values().index[0]
            mapped_class = str(nearest_major)
            distance_to_mapped = float(distances.loc[class_name, mapped_class])
        else:
            mapped_class = class_name
            distance_to_mapped = 0.0

        rows.append(
            {
                "class": class_name,
                "count": int(count),
                "is_rare": bool(is_rare),
                "mapped_class": mapped_class,
                "distance_to_mapped": distance_to_mapped,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_rare", "count"], ascending=[False, True])


def train_models_with_merged_target(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> pd.DataFrame:
    relief_features = get_relief_features(df)
    model_specs = build_model_specs(relief_features)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[MERGED_TARGET_COL])

    train_index, test_index = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    y_train = y[df.index.get_indexer(train_index)]
    y_test = y[df.index.get_indexer(test_index)]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    split_cols = [
        col for col in ID_COLS + [TARGET_COL, MERGED_TARGET_COL] if col in df.columns
    ]
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
            "target": MERGED_TARGET_COL,
            "original_target": TARGET_COL,
            "feature_columns": feature_cols,
            "classes": label_encoder.classes_.tolist(),
        }
        joblib.dump(model_payload, model_dir / f"{spec.name}.joblib")
        write_json(
            model_dir / "metrics.json",
            {
                "model": asdict(spec),
                "target": MERGED_TARGET_COL,
                "metrics": metrics,
                "xgboost_params": pipeline.named_steps["model"].get_params(),
            },
        )
        write_csv(report_df, model_dir / "classification_report.csv")
        write_csv(matrix_df, model_dir / "confusion_matrix.csv")
        save_feature_importance(pipeline, model_dir / "feature_importance.csv")

        predictions = df.loc[test_index, [col for col in ID_COLS if col in df.columns]].copy()
        predictions["true_soil_original"] = df.loc[test_index, TARGET_COL].to_numpy()
        predictions["true_soil_merged"] = label_encoder.inverse_transform(y_test)
        predictions["predicted_soil_merged"] = label_encoder.inverse_transform(y_pred)
        predictions["predicted_probability"] = y_proba.max(axis=1)
        for class_index, class_name in enumerate(label_encoder.classes_):
            predictions[f"prob_{class_name}"] = y_proba[:, class_index]
        write_csv(predictions, model_dir / "test_predictions.csv")

        summary_rows.append(
            {
                "model": spec.name,
                "description": spec.description,
                "target": MERGED_TARGET_COL,
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
    vector_numeric_features = relief_features
    vector_categorical_features = [
        col for col in TAXATION_FEATURES + SOIL_AUX_FEATURES if col in df.columns
    ]

    centroids, distances, nearest = build_class_vectors(
        df=df,
        numeric_features=vector_numeric_features,
        categorical_features=vector_categorical_features,
        distance_metric=args.distance_metric,
    )
    mapping_df = build_merge_mapping(
        df=df,
        distances=distances,
        min_class_count=args.min_class_count,
    )
    mapping = dict(zip(mapping_df["class"], mapping_df["mapped_class"]))
    df[MERGED_TARGET_COL] = df[TARGET_COL].map(mapping)

    write_csv(df, output_dir / "cleaned_soil_relief_tlu_merged.csv")
    write_json(output_dir / "data_profile.json", profile)
    write_csv(centroids, output_dir / "class_vectors.csv")
    write_csv(distances, output_dir / "class_vector_distances.csv")
    write_csv(nearest, output_dir / "class_vector_nearest_neighbors.csv")
    write_csv(mapping_df, output_dir / "class_merge_mapping.csv")

    model_specs = build_model_specs(relief_features)
    write_json(
        output_dir / "run_config.json",
        {
            "experiment": "merge_rare_classes_by_class_vectors",
            "input": str(input_path),
            "output": str(output_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_class_count": args.min_class_count,
            "distance_metric": args.distance_metric,
            "original_class_counts": df[TARGET_COL].value_counts().to_dict(),
            "merged_class_counts": df[MERGED_TARGET_COL].value_counts().to_dict(),
            "relief_features_used": relief_features,
            "vector_features_used": vector_numeric_features + vector_categorical_features,
            "models": [asdict(spec) for spec in model_specs],
        },
    )

    summary = train_models_with_merged_target(
        df=df,
        output_dir=output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )

    print("\nRare-class merge training complete.")
    print("\nClass merge mapping:")
    print(mapping_df.to_string(index=False))
    print("\nSummary metrics:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
