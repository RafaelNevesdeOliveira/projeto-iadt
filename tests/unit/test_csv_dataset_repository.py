from pathlib import Path

import pytest

from iadt_ml.domain.entities import Diagnosis
from iadt_ml.infrastructure.csv_dataset_repository import CsvDatasetRepository


def test_loads_features_labels_and_ignores_identifier_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "id,diagnosis,radius_mean,texture_mean,Unnamed: 32\n"
        "1,M,17.99,10.38,\n"
        "2,B,13.54,14.36,\n",
        encoding="utf-8",
    )

    snapshot = CsvDatasetRepository(csv_path).load()

    assert snapshot.feature_names == ("radius_mean", "texture_mean")
    assert snapshot.labels == (Diagnosis.MALIGNANT, Diagnosis.BENIGN)
    assert snapshot.rows == ((17.99, 10.38), (13.54, 14.36))


def test_coerces_missing_numeric_values_to_nan(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(
        "diagnosis,radius_mean,texture_mean\nM,17.99,\n",
        encoding="utf-8",
    )

    snapshot = CsvDatasetRepository(csv_path).load()

    assert len(snapshot.rows) == 1
    assert snapshot.rows[0][0] == 17.99
    assert snapshot.rows[0][1] != snapshot.rows[0][1]


def test_raises_when_diagnosis_column_is_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("id,radius_mean\n1,12.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="diagnosis"):
        CsvDatasetRepository(csv_path).load()
