from pathlib import Path

import joblib

from iadt_ml.application.use_cases import TrainModels
from iadt_ml.infrastructure.csv_dataset_repository import CsvDatasetRepository
from iadt_ml.infrastructure.joblib_model_repository import JoblibModelRepository
from iadt_ml.infrastructure.sklearn_dataset_partitioner import SklearnDatasetPartitioner
from iadt_ml.infrastructure.sklearn_evaluation_service import SklearnEvaluationService
from iadt_ml.infrastructure.sklearn_training_service import SklearnTrainingService

FIXTURE_PATH = Path("tests/fixtures/breast_cancer_sample.csv")


def test_trains_and_persists_selected_model(tmp_path: Path) -> None:
    model_path = tmp_path / "best_model.joblib"
    use_case = TrainModels(
        dataset_repository=CsvDatasetRepository(FIXTURE_PATH),
        dataset_partitioner=SklearnDatasetPartitioner(),
        training_service=SklearnTrainingService(),
        evaluation_service=SklearnEvaluationService(),
        model_repository=JoblibModelRepository(),
    )

    report = use_case.execute(model_path)

    assert len(report.artifacts) == 2
    assert report.selected_model.name in {artifact.name for artifact in report.artifacts}
    assert report.selected_model.test_evaluation is not None
    assert model_path.exists()
    persisted = joblib.load(model_path)
    assert persisted.name == report.selected_model.name
    assert hasattr(persisted.pipeline, "predict")


def test_selected_model_maximizes_recall_then_f1_then_accuracy(tmp_path: Path) -> None:
    model_path = tmp_path / "best_model.joblib"
    report = TrainModels(
        dataset_repository=CsvDatasetRepository(FIXTURE_PATH),
        dataset_partitioner=SklearnDatasetPartitioner(),
        training_service=SklearnTrainingService(),
        evaluation_service=SklearnEvaluationService(),
        model_repository=JoblibModelRepository(),
    ).execute(model_path)

    expected = max(
        report.artifacts,
        key=lambda artifact: (
            artifact.metrics.recall,
            artifact.metrics.f1_score,
            artifact.metrics.accuracy,
        ),
    )
    assert report.selected_model.name == expected.name
