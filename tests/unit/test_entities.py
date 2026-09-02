import pytest

from iadt_ml.domain.entities import DatasetSnapshot, Diagnosis


def test_dataset_snapshot_accepts_consistent_rows_and_labels() -> None:
    snapshot = DatasetSnapshot(
        feature_names=("radius", "texture"),
        rows=((1.0, 2.0), (3.0, 4.0)),
        labels=(Diagnosis.BENIGN, Diagnosis.MALIGNANT),
    )

    assert snapshot.feature_names == ("radius", "texture")
    assert len(snapshot.rows) == 2


def test_dataset_snapshot_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        DatasetSnapshot(feature_names=("radius",), rows=(), labels=())


def test_dataset_snapshot_rejects_mismatched_label_count() -> None:
    with pytest.raises(ValueError, match="same length"):
        DatasetSnapshot(
            feature_names=("radius",),
            rows=((1.0,), (2.0,)),
            labels=(Diagnosis.BENIGN,),
        )


def test_dataset_snapshot_rejects_row_with_wrong_feature_count() -> None:
    with pytest.raises(ValueError, match="feature count"):
        DatasetSnapshot(
            feature_names=("radius", "texture"),
            rows=((1.0,),),
            labels=(Diagnosis.BENIGN,),
        )
