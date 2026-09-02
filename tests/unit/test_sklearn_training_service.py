from pathlib import Path

from iadt_ml.infrastructure.csv_dataset_repository import CsvDatasetRepository
from iadt_ml.infrastructure.sklearn_training_service import (
    LOGISTIC_REGRESSION_NAME,
    RANDOM_FOREST_NAME,
    SklearnTrainingService,
)

FIXTURE_PATH = Path("tests/fixtures/breast_cancer_sample.csv")


def test_trains_logistic_regression_and_random_forest() -> None:
    dataset = CsvDatasetRepository(FIXTURE_PATH).load()

    artifacts = SklearnTrainingService().train(dataset)

    names = {artifact.name for artifact in artifacts}
    assert names == {LOGISTIC_REGRESSION_NAME, RANDOM_FOREST_NAME}
    for artifact in artifacts:
        assert 0.0 <= artifact.metrics.accuracy <= 1.0
        assert 0.0 <= artifact.metrics.recall <= 1.0
        assert 0.0 <= artifact.metrics.f1_score <= 1.0
        assert hasattr(artifact.pipeline, "predict")
