from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data" / "prediction"
DEFAULT_MODEL_ROOT = ROOT / "soil_xgboost_multiscale_merge_optuna_outputs"
DEFAULT_OUTPUT_DIR = ROOT / "soil_map_predictions"

MODEL_BY_FILE_TOKEN = {
    "model1": "model_1_relief_tlu",
    "model_1": "model_1_relief_tlu",
    "model2": "model_2_relief_only",
    "model_2": "model_2_relief_only",
    "model3": "model_3_relief_tlu_gran_mp",
    "model_3": "model_3_relief_tlu_gran_mp",
}

CATEGORY_FILL = {
    "TLU": "__MISSING_TLU__",
    "GRAN": "__MISSING_GRAN__",
    "MP": "__MISSING_MP__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Soil_pred for QGIS point CSV files using the trained "
            "soil classification pipelines."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_predicted.csv files in the output directory.",
    )
    return parser.parse_args()


def canonical_key(name: Any) -> str:
    text = str(name)
    text = text.replace("\ufeff", "")
    text = text.replace("\r", "")
    text = text.replace("\n", "")
    text = text.replace('"', "")
    text = text.strip().replace(" ", "")
    return text.lower()


def build_column_aliases() -> dict[str, str]:
    expected = [
        "WKT",
        "fid",
        "height_1",
        "slope_1",
        "aspect_1",
        "aspect_sin",
        "aspect_cos",
        "profile_1",
        "tangential_1",
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
        "TLU",
        "GRAN",
        "MP",
        "x_coord",
        "y_coord",
    ]
    aliases = {canonical_key(col): col for col in expected}
    aliases.update(
        {
            "тлу": "TLU",
            "tlutlu": "TLU",
            "tluтлу": "TLU",
            "x_cord": "x_coord",
            "tri1": "TRI_1",
            "tri_1": "TRI_1",
            "tpi3x31": "tpi_3x3_1",
            "tpi_3x31": "tpi_3x3_1",
            "tpi_3x3_1": "tpi_3x3_1",
            "rough15x151": "rough_15x15_1",
            "rough_15x151": "rough_15x15_1",
            "rough_15x15_1": "rough_15x15_1",
            "reliefstd3x31": "relief_std_3x3_1",
            "relief_std_3x31": "relief_std_3x3_1",
            "relief_std_3x3_1": "relief_std_3x3_1",
            # QGIS renamed the second distance field. In the training table this is osm_water_1.
            "dist_stream_1_2": "osm_water_1",
        }
    )
    return aliases


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    aliases = build_column_aliases()
    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()

    for col in df.columns:
        key = canonical_key(col)
        target = aliases.get(key)
        if target is None:
            continue
        if target in used_targets:
            continue
        if target in df.columns and col != target:
            continue
        rename_map[str(col)] = target
        used_targets.add(target)

    return df.rename(columns=rename_map), rename_map


def clean_string_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})
    return cleaned


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    missing_before_aspect = [col for col in feature_columns if col not in df.columns]

    if ("aspect_sin" in missing_before_aspect or "aspect_cos" in missing_before_aspect) and "aspect_1" in df.columns:
        aspect = pd.to_numeric(df["aspect_1"], errors="coerce") % 360.0
        radians = np.deg2rad(aspect)
        df["aspect_sin"] = np.sin(radians)
        df["aspect_cos"] = np.cos(radians)

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns after renaming: {missing}")

    features = df[feature_columns].copy()
    for col in feature_columns:
        if col in CATEGORY_FILL:
            features[col] = clean_string_column(features[col]).fillna(CATEGORY_FILL[col])
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    return features, missing


def infer_model_name(csv_path: Path) -> str:
    name = csv_path.stem.lower()
    for token, model_name in MODEL_BY_FILE_TOKEN.items():
        if token in name:
            return model_name
    raise ValueError(
        f"Cannot infer model name from file name: {csv_path.name}. "
        "Use names containing model1, model2, or model3."
    )


def load_payload(model_root: Path, model_name: str) -> dict[str, Any]:
    model_path = model_root / model_name / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "pipeline" not in payload or "label_encoder" not in payload:
        raise TypeError(f"Unexpected model payload format: {model_path}")
    return payload


def predict_file(csv_path: Path, model_root: Path, output_dir: Path, encoding: str, overwrite: bool) -> dict[str, Any]:
    model_name = infer_model_name(csv_path)
    payload = load_payload(model_root, model_name)
    pipeline = payload["pipeline"]
    label_encoder = payload["label_encoder"]
    feature_columns = list(payload["feature_columns"])

    df_raw = pd.read_csv(csv_path, encoding=encoding)
    df, rename_map = standardize_columns(df_raw)
    features, _ = prepare_features(df, feature_columns)

    pred_codes = pipeline.predict(features)
    pred_labels = label_encoder.inverse_transform(pred_codes.astype(int))
    probabilities = pipeline.predict_proba(features)
    max_probability = probabilities.max(axis=1)

    result = df_raw.copy()
    result["Soil_pred"] = pred_labels
    result["Soil_pred_code"] = pred_codes.astype(int)
    result["Soil_pred_proba"] = max_probability
    result["Soil_model"] = model_name

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_path.stem}_predicted.csv"
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists, use --overwrite: {out_path}")
        out_path.unlink()
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    counts = pd.Series(pred_labels).value_counts().sort_index().to_dict()
    return {
        "input": str(csv_path),
        "output": str(out_path),
        "model": model_name,
        "rows": int(len(result)),
        "feature_columns": feature_columns,
        "renamed_columns": rename_map,
        "prediction_counts": {str(key): int(value) for key, value in counts.items()},
        "missing_numeric_values_used_imputer": {
            col: int(features[col].isna().sum())
            for col in feature_columns
            if col not in CATEGORY_FILL and int(features[col].isna().sum()) > 0
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_root = args.model_root

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if not model_root.exists():
        raise FileNotFoundError(model_root)

    csv_files = sorted(
        path for path in input_dir.glob(args.pattern) if not path.name.lower().endswith("_predicted.csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} by pattern {args.pattern!r}")

    reports = [
        predict_file(path, model_root, output_dir, args.encoding, args.overwrite)
        for path in csv_files
    ]

    report_path = output_dir / "prediction_run_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Prediction complete.")
    print(f"Model root: {model_root}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    for report in reports:
        print(f"- {Path(report['input']).name} -> {Path(report['output']).name}: {report['rows']} rows, {report['model']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
