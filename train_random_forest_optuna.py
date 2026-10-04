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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from train_xgboost_smote import class_counts, make_sampler
from train_xgboost_soil import (
    ID_COLS,
    RANDOM_STATE_DEFAULT,
    SOURCE_DEFAULT,
    TARGET_COL,
    ModelSpec,
    build_model_specs,
    clean_data,
    evaluate_predictions,
    get_relief_features,
    save_feature_importance,
    write_csv,
    write_json,
)


SOURCE_MULTISCALE_DEFAULT = str(Path(__file__).resolve().parent / "data/training/soil_full_osm_multiscale.csv")
OUTPUT_DEFAULT = "soil_random_forest_8classes_optuna_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train three RandomForestClassifier soil models for original 8 classes "
            "with Optuna hyperparameter tuning, optionally with SMOTE/SMOTENC."
        )
    )
    parser.add_argument("--input", default=SOURCE_MULTISCALE_DEFAULT or SOURCE_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--rf-n-estimators-min", type=int, default=100)
    parser.add_argument("--rf-n-estimators-max", type=int, default=700)
    parser.add_argument("--rf-n-estimators-step", type=int, default=100)
    parser.add_argument("--rf-max-depth-max", type=int, default=32)
    parser.add_argument(
        "--optimization-metric",
        choices=["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"],
        default="macro_f1",
    )
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--smote-target-count", type=int, default=10000)
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--save-pickle", action="store_true")
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                spec.categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_rf_pipeline(spec: ModelSpec, random_state: int, params: dict[str, Any], n_jobs: int = -1) -> Pipeline:
    classifier = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
        **params,
    )
    return Pipeline(steps=[("preprocess", make_preprocess(spec)), ("model", classifier)])


def suggest_rf_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    max_depth_choice = trial.suggest_categorical(
        "max_depth_choice", ["none", "depth"]
    )
    max_depth = None
    if max_depth_choice == "depth":
        max_depth = trial.suggest_int("max_depth", 4, args.rf_max_depth_max, step=2)

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            args.rf_n_estimators_min,
            args.rf_n_estimators_max,
            step=args.rf_n_estimators_step,
        ),
        "max_depth": max_depth,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.75, None]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }
    if not args.use_smote:
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        )
    else:
        params["class_weight"] = None
    return params


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
        pipeline.fit(X_train, y_train)
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
    X_opt_train = df.loc[opt_train_index, feature_cols].copy()
    X_valid = df.loc[valid_index, feature_cols].copy()
    y_opt_train = y_all[df.index.get_indexer(opt_train_index)]
    y_valid = y_all[df.index.get_indexer(valid_index)]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_rf_params(trial, args=args)
        pipeline = make_rf_pipeline(spec=spec, random_state=args.random_state, params=params, n_jobs=args.n_jobs)
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
    best_params.pop("max_depth_choice", None)
    final_pipeline = make_rf_pipeline(
        spec=spec,
        random_state=args.random_state,
        params=best_params,
        n_jobs=args.n_jobs,
    )
    y_resampled, sampler_report = fit_with_optional_smote(
        pipeline=final_pipeline,
        X_train=X_train_full,
        y_train=y_train_full,
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
    predictions["true_soil"] = label_encoder.inverse_transform(y_test)
    predictions["predicted_soil"] = label_encoder.inverse_transform(y_pred)
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
        "target": TARGET_COL,
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
            "target": TARGET_COL,
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
        "target": TARGET_COL,
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
) -> None:
    models: dict[str, Any] = {}
    for spec in model_specs:
        models[spec.name] = joblib.load(output_dir / spec.name / f"{spec.name}.joblib")

    with (output_dir / "label_encoder.pkl").open("wb") as file:
        pickle.dump(label_encoder, file, protocol=pickle.HIGHEST_PROTOCOL)

    bundle = {
        "experiment": "random_forest_original_8_classes_optuna",
        "target": TARGET_COL,
        "models": models,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "summary_metrics": summary.to_dict(orient="records"),
    }
    with (output_dir / "soil_random_forest_models_bundle.pkl").open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, profile = clean_data(input_path)
    relief_features = get_relief_features(df)
    model_specs = build_model_specs(relief_features)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL])

    train_index, test_index = train_test_split(
        df.index,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    split_cols = [col for col in dict.fromkeys(ID_COLS + [TARGET_COL]) if col in df.columns]
    write_csv(df.loc[train_index, split_cols], output_dir / "splits" / "train_rows.csv")
    write_csv(df.loc[test_index, split_cols], output_dir / "splits" / "test_rows.csv")
    write_csv(df, output_dir / "cleaned_soil_relief_tlu.csv")
    write_json(output_dir / "data_profile.json", profile)
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    write_json(
        output_dir / "run_config.json",
        {
            "experiment": "random_forest_original_8_classes_optuna",
            "input": str(input_path),
            "output": str(output_dir),
            "target": TARGET_COL,
            "test_size": args.test_size,
            "valid_size_inside_train": args.valid_size,
            "random_state": args.random_state,
            "n_trials": args.n_trials,
            "n_jobs": args.n_jobs,
            "timeout": args.timeout,
            "rf_n_estimators_min": args.rf_n_estimators_min,
            "rf_n_estimators_max": args.rf_n_estimators_max,
            "rf_n_estimators_step": args.rf_n_estimators_step,
            "rf_max_depth_max": args.rf_max_depth_max,
            "optimization_metric": args.optimization_metric,
            "use_smote": bool(args.use_smote),
            "smote_target_count": args.smote_target_count if args.use_smote else None,
            "save_pickle": bool(args.save_pickle),
            "class_counts": df[TARGET_COL].value_counts().to_dict(),
            "relief_features_used": relief_features,
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
            output_dir=output_dir,
            args=args,
        )
        for spec in model_specs
    ]
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    write_csv(summary, output_dir / "summary_metrics.csv")
    if args.save_pickle:
        write_pickle_bundle(output_dir, model_specs, label_encoder, summary)

    print("\nRandom Forest 8-class training complete.")
    print("\nSummary metrics:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
