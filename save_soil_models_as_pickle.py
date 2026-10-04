from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import joblib


MODEL_NAMES = [
    "model_1_relief_tlu",
    "model_2_relief_only",
    "model_3_relief_tlu_gran_mp",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_records(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved soil XGBoost .joblib artifacts into pickle .pkl files."
    )
    parser.add_argument("output_dir", help="Existing training output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    label_encoder_path = output_dir / "label_encoder.joblib"
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Missing label encoder: {label_encoder_path}")

    label_encoder = joblib.load(label_encoder_path)
    models: dict[str, Any] = {}
    saved_paths: list[Path] = []

    for model_name in MODEL_NAMES:
        joblib_path = output_dir / model_name / f"{model_name}.joblib"
        if not joblib_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {joblib_path}")
        payload = joblib.load(joblib_path)
        models[model_name] = payload

        pkl_path = output_dir / model_name / f"{model_name}.pkl"
        with pkl_path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        saved_paths.append(pkl_path)

    label_pkl_path = output_dir / "label_encoder.pkl"
    with label_pkl_path.open("wb") as file:
        pickle.dump(label_encoder, file, protocol=pickle.HIGHEST_PROTOCOL)
    saved_paths.append(label_pkl_path)

    run_config = read_json(output_dir / "run_config.json")
    data_profile = read_json(output_dir / "data_profile.json")
    summary_metrics = read_csv_records(output_dir / "summary_metrics.csv")
    class_merge_mapping = read_csv_records(output_dir / "class_merge_mapping.csv")
    original_class_mapping = read_csv_records(output_dir / "class_mapping_original_8_classes.csv")

    bundle = {
        "source_output_dir": str(output_dir.resolve()),
        "target": next(iter(models.values())).get("target") if models else None,
        "models": models,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "summary_metrics": summary_metrics,
        "run_config": run_config,
        "data_profile": data_profile,
        "class_merge_mapping": class_merge_mapping,
        "class_mapping_original_8_classes": original_class_mapping,
        "notes": [
            "Each model payload contains a fitted sklearn Pipeline under key 'pipeline'.",
            "Use payload['label_encoder'].inverse_transform(...) to convert numeric predictions to soil class names.",
        ],
    }
    bundle_path = output_dir / "soil_models_bundle.pkl"
    with bundle_path.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)
    saved_paths.append(bundle_path)

    print("Saved pickle artifacts:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
