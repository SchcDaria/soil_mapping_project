from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_experiments() -> dict:
    return json.loads((ROOT / "configs" / "experiments.json").read_text(encoding="utf-8"))


def build_command(name: str, experiment: dict, output: Path, *, smoke: bool = False,
                  input_path: Path | None = None, n_trials: int | None = None) -> list[str]:
    source = input_path.resolve() if input_path else ROOT / experiment["input"]
    if not source.is_file():
        raise ValueError(f"Training CSV does not exist: {source}")
    options = dict(experiment["options"])
    if n_trials is not None:
        if "optuna" not in experiment["script"] or n_trials < 1:
            raise ValueError("--n-trials requires an Optuna experiment and a positive number")
        options["n_trials"] = n_trials
    if smoke:
        if experiment["script"] != "train_random_forest_optuna.py":
            raise ValueError("--smoke is supported for rf8 and rf8_smote")
        options.update(n_trials=1, rf_n_estimators_min=2, rf_n_estimators_max=2,
                       rf_n_estimators_step=1, n_jobs=1)
        if options.get("use_smote"):
            options["smote_target_count"] = 300
    command = [sys.executable, "-B", str(ROOT / experiment["script"]),
               "--input", str(source), "--output", str(output)]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif value is not False and value is not None:
            command.extend([flag, str(value)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a saved soil-classification experiment.")
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-trials", type=int)
    args = parser.parse_args()
    experiments = load_experiments()
    if args.list:
        for name, config in experiments.items():
            status = "saved" if config["historical_results_available"] else "not saved"
            print(f"{name}: {config['script']}; {Path(config['input']).name}; {status}")
        return 0
    if args.experiment not in experiments:
        parser.error("Choose an experiment from --list")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output.resolve() if args.output else ROOT / "runs" / f"{args.experiment}_{stamp}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output already contains files; choose a new directory: {output}")
    try:
        command = build_command(args.experiment, experiments[args.experiment], output,
                                smoke=args.smoke, input_path=args.input, n_trials=args.n_trials)
    except ValueError as error:
        parser.error(str(error))
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
