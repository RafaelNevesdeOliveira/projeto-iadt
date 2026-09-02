from pathlib import Path

import joblib

from iadt_ml.presentation import cli

FIXTURE_PATH = Path("tests/fixtures/breast_cancer_sample.csv")


def test_build_parser_requires_train_dataset() -> None:
    parser = cli.build_parser()
    arguments = parser.parse_args(["train", "--dataset", "data/raw/breast_cancer.csv"])

    assert arguments.command == "train"
    assert arguments.dataset == Path("data/raw/breast_cancer.csv")
    assert arguments.output == cli.DEFAULT_MODEL_PATH


def test_main_trains_and_prints_report(tmp_path: Path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "models" / "best_model.joblib"
    monkeypatch.setattr(
        "sys.argv",
        ["iadt-ml", "train", "--dataset", str(FIXTURE_PATH), "--output", str(output_path)],
    )

    cli.main()

    captured = capsys.readouterr().out
    assert "logistic_regression:" in captured
    assert "random_forest:" in captured
    assert "selected_model=" in captured
    assert f"model_path={output_path}" in captured
    assert output_path.exists()
    assert joblib.load(output_path).name in {"logistic_regression", "random_forest"}
