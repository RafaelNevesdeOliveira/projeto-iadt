from pathlib import Path

import pytest

from iadt_ml.application.use_cases import TrainModels
from iadt_ml.domain.entities import (
    ConfusionMatrix,
    DatasetPartition,
    DatasetSnapshot,
    Diagnosis,
    EvaluationMetrics,
    ModelArtifact,
    ModelEvaluation,
)


class FakeDatasetRepository:
    def load(self) -> DatasetSnapshot:
        return DatasetSnapshot(
            feature_names=("radius",),
            rows=((1.0,), (2.0,)),
            labels=(Diagnosis.BENIGN, Diagnosis.MALIGNANT),
        )


class FakeTrainingService:
    def __init__(self, artifacts: tuple[ModelArtifact, ...]) -> None:
        self._artifacts = artifacts

    def train(self, dataset: DatasetSnapshot) -> tuple[ModelArtifact, ...]:
        return self._artifacts


class FakeDatasetPartitioner:
    def split(self, dataset: DatasetSnapshot) -> DatasetPartition:
        return DatasetPartition(training=dataset, testing=dataset)


class FakeEvaluationService:
    def evaluate(self, artifact: ModelArtifact, dataset: DatasetSnapshot) -> ModelEvaluation:
        return ModelEvaluation(
            metrics=EvaluationMetrics(accuracy=0.90, recall=0.90, f1_score=0.90, roc_auc=0.90),
            confusion_matrix=ConfusionMatrix(1, 0, 0, 1),
        )


class FakeModelRepository:
    def __init__(self) -> None:
        self.saved_artifact: ModelArtifact | None = None

    def save(self, artifact: ModelArtifact, destination: Path) -> Path:
        self.saved_artifact = artifact
        return destination


def _artifact(name: str, accuracy: float, recall: float, f1_score: float) -> ModelArtifact:
    return ModelArtifact(name, object(), EvaluationMetrics(accuracy, recall, f1_score))


def _use_case(training_service: FakeTrainingService, repository: FakeModelRepository) -> TrainModels:
    return TrainModels(
        FakeDatasetRepository(),
        FakeDatasetPartitioner(),
        training_service,
        FakeEvaluationService(),
        repository,
    )


def test_selects_model_with_highest_malignant_recall() -> None:
    repository = FakeModelRepository()
    training_service = FakeTrainingService(
        (
            _artifact("baseline", 0.90, 0.80, 0.80),
            _artifact("candidate", 0.85, 0.90, 0.75),
        )
    )
    use_case = _use_case(training_service, repository)

    report = use_case.execute(Path("models/best_model.joblib"))

    assert report.selected_model.name == "candidate"
    assert repository.saved_artifact == report.selected_model


def test_breaks_recall_ties_using_f1_then_accuracy() -> None:
    repository = FakeModelRepository()
    training_service = FakeTrainingService(
        (
            _artifact("lower_f1", 0.99, 0.90, 0.80),
            _artifact("higher_f1", 0.80, 0.90, 0.88),
        )
    )
    use_case = _use_case(training_service, repository)

    report = use_case.execute(Path("models/best_model.joblib"))

    assert report.selected_model.name == "higher_f1"


def test_breaks_recall_and_f1_ties_using_accuracy() -> None:
    repository = FakeModelRepository()
    training_service = FakeTrainingService(
        (
            _artifact("lower_accuracy", 0.80, 0.90, 0.85),
            _artifact("higher_accuracy", 0.95, 0.90, 0.85),
        )
    )
    use_case = _use_case(training_service, repository)

    report = use_case.execute(Path("models/best_model.joblib"))

    assert report.selected_model.name == "higher_accuracy"


def test_raises_when_training_produces_no_artifacts() -> None:
    use_case = _use_case(FakeTrainingService(()), FakeModelRepository())

    with pytest.raises(ValueError, match="at least one model"):
        use_case.execute(Path("models/best_model.joblib"))
