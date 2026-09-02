from pathlib import Path

from iadt_ml.application.use_cases import AnalyzeDataset, ExplainSelectedModel
from iadt_ml.domain.entities import (
    DatasetPartition,
    DatasetSnapshot,
    Diagnosis,
    EvaluationMetrics,
    ModelArtifact,
)


class FakeDatasetRepository:
    def load(self) -> DatasetSnapshot:
        return DatasetSnapshot(
            feature_names=("radius",),
            rows=((1.0,), (2.0,)),
            labels=(Diagnosis.BENIGN, Diagnosis.MALIGNANT),
        )


class FakePartitioner:
    def split(self, dataset: DatasetSnapshot) -> DatasetPartition:
        return DatasetPartition(training=dataset, testing=dataset)


class FakeModelRepository:
    def load(self, source: Path) -> ModelArtifact:
        return ModelArtifact("model", object(), EvaluationMetrics(0.9, 0.9, 0.9))


class FakeAnalysisService:
    def __init__(self) -> None:
        self.received_destination: Path | None = None

    def analyze(self, dataset: DatasetSnapshot, destination: Path) -> None:
        self.received_destination = destination


class FakeExplanationService:
    def __init__(self) -> None:
        self.received_destination: Path | None = None

    def explain(self, artifact: ModelArtifact, dataset: DatasetSnapshot, destination: Path) -> None:
        self.received_destination = destination


def test_analyzes_dataset_through_port() -> None:
    service = FakeAnalysisService()
    destination = Path("reports/figures")

    AnalyzeDataset(FakeDatasetRepository(), service).execute(destination)

    assert service.received_destination == destination


def test_explains_selected_model_through_ports() -> None:
    service = FakeExplanationService()
    destination = Path("reports/figures")

    ExplainSelectedModel(
        FakeDatasetRepository(),
        FakePartitioner(),
        FakeModelRepository(),
        service,
    ).execute(Path("models/best_model.joblib"), destination)

    assert service.received_destination == destination
