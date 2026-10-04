from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from xml.etree import ElementTree

from run_experiment import ROOT, build_command, load_experiments, main


class ProjectTests(unittest.TestCase):
    def test_all_experiment_options_are_accepted_by_training_scripts(self):
        for name, config in load_experiments().items():
            with self.subTest(name=name):
                command = build_command(name, config, ROOT / "runs" / "test")
                module = importlib.import_module(Path(config["script"]).stem)
                with patch.object(sys, "argv", [command[2], *command[3:]]):
                    args = module.parse_args()
                self.assertTrue(Path(args.input).is_file())
                self.assertEqual(args.test_size, 0.25)

    def test_refuses_to_overwrite_an_existing_result(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            path = Path(directory)
            (path / "result.txt").write_text("keep", encoding="ascii")
            with patch.object(sys, "argv", ["run_experiment.py", "rf8", "--output", str(path)]):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual((path / "result.txt").read_text(encoding="ascii"), "keep")

    def test_dry_run_does_not_create_a_result_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            path = Path(directory) / "untouched"
            with patch.object(sys, "argv", ["run_experiment.py", "rf8", "--dry-run", "--output", str(path)]):
                self.assertEqual(main(), 0)
            self.assertFalse(path.exists())

    def test_smoke_retains_three_model_training_and_caps_resources(self):
        config = load_experiments()["rf8_smote"]
        command = build_command("rf8_smote", config, ROOT / "runs" / "test", smoke=True)
        import train_random_forest_optuna as training
        with patch.object(sys, "argv", [command[2], *command[3:]]):
            args = training.parse_args()
        self.assertEqual((args.n_trials, args.rf_n_estimators_max, args.n_jobs), (1, 2, 1))
        self.assertEqual(args.smote_target_count, 300)
        self.assertTrue(args.use_smote)

    def test_qgis_keeps_every_layer_and_its_reference(self):
        with ZipFile(ROOT / "gis/diplom_mag.qgz") as archive:
            self.assertIsNone(archive.testzip())
            project = ElementTree.fromstring(archive.read("diplom_mag.qgs"))
        layers = project.findall("./projectlayers/maplayer")
        self.assertEqual(len(layers), 147)
        self.assertTrue(all(layer.findtext("datasource") for layer in layers))

    def test_final_dataset_preserves_eight_classes_and_three_feature_sets(self):
        from train_xgboost_soil import build_model_specs, clean_data, get_relief_features

        data, profile = clean_data(ROOT / "data/training/soil_full_osm_multiscale.csv")
        self.assertEqual((len(data), profile["dropped_rows"]), (1358, 0))
        self.assertEqual(data["Soil"].nunique(), 8)
        self.assertEqual(profile["class_counts"]["т1п пов огл"], 14)
        specs = build_model_specs(get_relief_features(data))
        self.assertEqual([len(spec.numeric_features) for spec in specs], [25, 25, 25])
        self.assertEqual([spec.categorical_features for spec in specs],
                         [["TLU"], [], ["TLU", "GRAN", "MP"]])
        for spec in specs:
            self.assertTrue({"Soil", "fid", "WKT", "x", "y"}.isdisjoint(
                spec.numeric_features + spec.categorical_features))

    def test_imputer_reuses_training_median_for_new_points(self):
        import numpy as np
        import pandas as pd
        from train_random_forest_optuna import make_preprocess
        from train_xgboost_soil import ModelSpec

        preprocess = make_preprocess(ModelSpec("test", "", ["height_1"], ["TLU"]))
        train = pd.DataFrame({"height_1": [10, 20, 30, np.nan], "TLU": ["A", "A", "B", "B"]})
        preprocess.fit(train)
        new_points = pd.DataFrame({"height_1": [np.nan, 10000], "TLU": ["C", "A"]})
        transformed = preprocess.transform(new_points)
        self.assertEqual(preprocess.named_transformers_["numeric"]["imputer"].statistics_[0], 20)
        np.testing.assert_array_equal(transformed[:, 0], [20, 10000])

    @classmethod
    def setUpClass(cls):
        (ROOT / "runs").mkdir(exist_ok=True)


if __name__ == "__main__":
    unittest.main()
