from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from train_xgboost_merge_rare import (
    MERGED_TARGET_COL,
    build_class_vectors,
    build_merge_mapping,
)
from train_xgboost_smote import class_counts, make_sampler
from train_xgboost_soil import (
    ID_COLS,
    RANDOM_STATE_DEFAULT,
    SOIL_AUX_FEATURES,
    TARGET_COL,
    TAXATION_FEATURES,
    ModelSpec,
    build_model_specs,
    clean_data,
    evaluate_predictions,
    get_relief_features,
    save_feature_importance,
    write_csv,
    write_json,
)


OSM_WATER_SOURCE_DEFAULT = (
    str(Path(__file__).resolve().parent / "data/training/soil_relief_extended_full_osm_water.csv")
)
OUTPUT_DEFAULT = "soil_xgboost_osm_merge_optuna_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge rare soil classes into 7 classes, optionally apply SMOTE/SMOTENC, "
            "then tune XGBoost classifiers with Optuna."
        )
    )
    parser.add_argument("--input", default=OSM_WATER_SOURCE_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--min-class-count", type=int, default=30)
    parser.add_argument(
        "--no-merge-rare",
        action="store_true",
        help="Use original Soil classes without merging rare classes. This keeps 8 classes.",
    )
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument(
        "--optimization-metric",
        choices=["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"],
        default="macro_f1",
    )
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--smote-target-count", type=int, default=10000)
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument(
        "--save-pickle",
        action="store_true",
        help="Also save fitted model payloads as .pkl files and one combined bundle.",
    )
    return parser.parse_args()


def make_preprocess(spec: ModelSpec) -> ColumnTransformer:
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

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_optuna_pipeline(
    spec: ModelSpec,
    num_classes: int,
    random_state: int,
    params: dict[str, Any],
) -> Pipeline:
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        **params,
    )
    return Pipeline(steps=[("preprocess", make_preprocess(spec)), ("model", classifier)])


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
    }


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, average="weighted"))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric}")


def make_equal_smote_strategy(y_train: np.ndarray, target_count: int) -> dict[int, int]:
    counts = pd.Series(y_train).value_counts()
    too_large = counts[counts > target_count]
    if not too_large.empty:
        raise ValueError(
            "SMOTE target count must be greater than or equal to every class count. "
            f"Current counts above target {target_count}: {too_large.to_dict()}"
        )
    return {int(class_id): int(target_count) for class_id in counts.index}


def fit_with_optional_smote(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_features: list[str],
    use_smote: bool,
    smote_target_count: int,
    smote_k_neighbors: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not use_smote:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        return y_train, None

    sampling_strategy = make_equal_smote_strategy(y_train, smote_target_count)
    sampler, sampler_report = make_sampler(
        X_train=X_train,
        categorical_features=categorical_features,
        y_train=y_train,
        sampling_strategy=sampling_strategy,
        requested_k_neighbors=smote_k_neighbors,
        random_state=random_state,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    pipeline.fit(X_resampled, y_resampled)
    return y_resampled, {
        **sampler_report,
        "smote_target_count": smote_target_count,
        "sampling_strategy": sampling_strategy,
    }


def tune_one_model(
    df: pd.DataFrame,
    spec: ModelSpec,
    train_index: pd.Index,
    test_index: pd.Index,
    y_all: np.ndarray,
    label_encoder: LabelEncoder,
    target_col: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_dir = output_dir / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = spec.numeric_features + spec.categorical_features
    X_train_full = df.loc[train_index, feature_cols].copy()
    X_test = df.loc[test_index, feature_cols].copy()
    y_train_full = y_all[df.index.get_indexer(train_index)]
    y_test = y_all[df.index.get_indexer(test_index)]

    opt_train_index, valid_index = train_test_split(
        train_index,
        test_size=args.valid_size,
        random_state=args.random_state,
        stratify=y_train_full,
    )
    y_opt_train = y_all[df.index.get_indexer(opt_train_index)]
    y_valid = y_all[df.index.get_indexer(valid_index)]
    X_opt_train = df.loc[opt_train_index, feature_cols].copy()
    X_valid = df.loc[valid_index, feature_cols].copy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        pipeline = make_optuna_pipeline(
            spec=spec,
            num_classes=len(label_encoder.classes_),
            random_state=args.random_state,
            params=params,
        )
        fit_with_optional_smote(
            pipeline=pipeline,
            X_train=X_opt_train,
            y_train=y_opt_train,
            categorical_features=spec.categorical_features,
            use_smote=args.use_smote,
            smote_target_count=args.smote_target_count,
            smote_k_neighbors=args.smote_k_neighbors,
            random_state=args.random_state,
        )
        y_pred = pipeline.predict(X_valid)
        return score_predictions(y_valid, y_pred, args.optimization_metric)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    best_params = dict(study.best_params)
    final_pipeline = make_optuna_pipeline(
        spec=spec,
        num_classes=len(label_encoder.classes_),
        random_state=args.random_state,
        params=best_params,
    )
    y_final_train = y_train_full
    y_resampled, sampler_report = fit_with_optional_smote(
        pipeline=final_pipeline,
        X_train=X_train_full,
        y_train=y_final_train,
        categorical_features=spec.categorical_features,
        use_smote=args.use_smote,
        smote_target_count=args.smote_target_count,
        smote_k_neighbors=args.smote_k_neighbors,
        random_state=args.random_state,
    )

    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)
    metrics, report_df, matrix_df = evaluate_predictions(y_test, y_pred, label_encoder.classes_)

    trials_df = study.trials_dataframe()
    write_csv(trials_df, model_dir / "optuna_trials.csv")
    write_csv(report_df, model_dir / "classification_report.csv")
    write_csv(matrix_df, model_dir / "confusion_matrix.csv")
    save_feature_importance(final_pipeline, model_dir / "feature_importance.csv")

    predictions = df.loc[test_index, [col for col in ID_COLS if col in df.columns]].copy()
    predictions["true_soil_original"] = df.loc[test_index, TARGET_COL].to_numpy()
    predictions[f"true_{target_col}"] = label_encoder.inverse_transform(y_test)
    predictions[f"predicted_{target_col}"] = label_encoder.inverse_transform(y_pred)
    predictions["predicted_probability"] = y_proba.max(axis=1)
    for class_index, class_name in enumerate(label_encoder.classes_):
        predictions[f"prob_{class_name}"] = y_proba[:, class_index]
    write_csv(predictions, model_dir / "test_predictions.csv")

    train_counts_before = class_counts(y_train_full, label_encoder.classes_)
    train_counts_after = class_counts(y_resampled, label_encoder.classes_)
    model_payload = {
        "pipeline": final_pipeline,
        "label_encoder": label_encoder,
        "model_spec": asdict(spec),
        "target": target_col,
        "original_target": TARGET_COL,
        "feature_columns": feature_cols,
        "classes": label_encoder.classes_.tolist(),
        "best_params": best_params,
        "best_validation_score": float(study.best_value),
        "optimization_metric": args.optimization_metric,
        "resampling": sampler_report,
    }
    joblib.dump(model_payload, model_dir / f"{spec.name}.joblib")
    if args.save_pickle:
        with (model_dir / f"{spec.name}.pkl").open("wb") as file:
            pickle.dump(model_payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        model_dir / "metrics.json",
        {
            "model": asdict(spec),
            "target": target_col,
            "metrics": metrics,
            "optimization_metric": args.optimization_metric,
            "best_validation_score": float(study.best_value),
            "best_params": best_params,
            "use_smote": bool(args.use_smote),
            "resampling": sampler_report,
            "train_class_counts_before": train_counts_before,
            "train_class_counts_after": train_counts_after,
        },
    )

    return {
        "model": spec.name,
        "description": spec.description,
        "target": target_col,
        "features": ", ".join(feature_cols),
        "use_smote": bool(args.use_smote),
        "smote_target_count": args.smote_target_count if args.use_smote else None,
        "n_trials": len(study.trials),
        "optimization_metric": args.optimization_metric,
        "best_validation_score": float(study.best_value),
        **metrics,
    }


def write_pickle_bundle(
    output_dir: Path,
    model_specs: list[ModelSpec],
    label_encoder: LabelEncoder,
    summary: pd.DataFrame,
    target_col: str,
    experiment_name: str,
) -> None:
    models: dict[str, Any] = {}
    for spec in model_specs:
        models[spec.name] = joblib.load(output_dir / spec.name / f"{spec.name}.joblib")

    with (output_dir / "label_encoder.pkl").open("wb") as file:
        pickle.dump(label_encoder, file, protocol=pickle.HIGHEST_PROTOCOL)

    bundle = {
        "experiment": experiment_name,
        "target": target_col,
        "models": models,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "summary_metrics": summary.to_dict(orient="records"),
        "notes": [
            "Each model payload contains a fitted sklearn Pipeline under key 'pipeline'.",
            "Use payload['label_encoder'].inverse_transform(...) to convert numeric predictions to soil class names.",
        ],
    }
    with (output_dir / "soil_models_bundle.pkl").open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)


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

    if args.no_merge_rare:
        target_col = TARGET_COL
        experiment_name = "original_8_classes_optuna"
        mapping_df = pd.DataFrame(
            {
                "class": df[TARGET_COL].value_counts().index,
                "count": df[TARGET_COL].value_counts().values,
                "is_rare": False,
                "mapped_class": df[TARGET_COL].value_counts().index,
                "distance_to_mapped": 0.0,
            }
        )
        write_csv(mapping_df, output_dir / "class_mapping_original_8_classes.csv")
        write_csv(df, output_dir / "cleaned_soil_relief_tlu.csv")
    else:
        target_col = MERGED_TARGET_COL
        experiment_name = "merge_rare_classes_optuna"
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
        write_csv(centroids, output_dir / "class_vectors.csv")
        write_csv(distances, output_dir / "class_vector_distances.csv")
        write_csv(nearest, output_dir / "class_vector_nearest_neighbors.csv")
        write_csv(mapping_df, output_dir / "class_merge_mapping.csv")

    write_json(output_dir / "data_profile.json", profile)

    model_specs = build_model_specs(relief_features)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])
    train_index, test_index = train_test_split(
        df.index,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    split_cols = [
        col for col in dict.fromkeys(ID_COLS + [TARGET_COL, target_col]) if col in df.columns
    ]
    write_csv(df.loc[train_index, split_cols], output_dir / "splits" / "train_rows.csv")
    write_csv(df.loc[test_index, split_cols], output_dir / "splits" / "test_rows.csv")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    write_json(
        output_dir / "run_config.json",
        {
            "experiment": experiment_name,
            "input": str(input_path),
            "output": str(output_dir),
            "test_size": args.test_size,
            "valid_size_inside_train": args.valid_size,
            "random_state": args.random_state,
            "n_trials": args.n_trials,
            "timeout": args.timeout,
            "min_class_count": args.min_class_count,
            "no_merge_rare": bool(args.no_merge_rare),
            "distance_metric": args.distance_metric,
            "optimization_metric": args.optimization_metric,
            "use_smote": bool(args.use_smote),
            "smote_target_count": args.smote_target_count if args.use_smote else None,
            "save_pickle": bool(args.save_pickle),
            "original_class_counts": df[TARGET_COL].value_counts().to_dict(),
            "target_class_counts": df[target_col].value_counts().to_dict(),
            "relief_features_used": relief_features,
            "vector_features_used": vector_numeric_features + vector_categorical_features,
            "models": [asdict(spec) for spec in model_specs],
        },
    )

    summary_rows = [
        tune_one_model(
            df=df,
            spec=spec,
            train_index=train_index,
            test_index=test_index,
            y_all=y,
            label_encoder=label_encoder,
            target_col=target_col,
            output_dir=output_dir,
            args=args,
        )
        for spec in model_specs
    ]
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    write_csv(summary, output_dir / "summary_metrics.csv")
    if args.save_pickle:
        write_pickle_bundle(
            output_dir=output_dir,
            model_specs=model_specs,
            label_encoder=label_encoder,
            summary=summary,
            target_col=target_col,
            experiment_name=experiment_name,
        )

    print(f"\n{experiment_name} training complete.")
    print("\nClass mapping:")
    print(mapping_df.to_string(index=False))
    print("\nSummary metrics:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
