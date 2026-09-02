from pathlib import Path
from typing import Protocol

from iadt_ml.domain.entities import (
    DatasetPartition,
    DatasetSnapshot,
    ModelArtifact,
    ModelEvaluation,
)


class DatasetRepository(Protocol):
    """Define como a aplicação obtém um dataset sem depender de CSV ou banco."""

    def load(self) -> DatasetSnapshot: ...


class ModelTrainingService(Protocol):
    """Define como candidatos de Machine Learning são treinados."""

    def train(self, dataset: DatasetSnapshot) -> tuple[ModelArtifact, ...]: ...


class DatasetPartitioner(Protocol):
    """Define como o dataset é separado entre treino e teste."""

    def split(self, dataset: DatasetSnapshot) -> DatasetPartition: ...


class ModelEvaluationService(Protocol):
    """Define como um modelo é avaliado em dados ainda não vistos."""

    def evaluate(self, artifact: ModelArtifact, dataset: DatasetSnapshot) -> ModelEvaluation: ...


class ModelRepository(Protocol):
    """Define como artefatos de modelo são salvos e recuperados."""

    def save(self, artifact: ModelArtifact, destination: Path) -> Path: ...

    def load(self, source: Path) -> ModelArtifact: ...


class DatasetAnalysisService(Protocol):
    """Define como tabelas e gráficos exploratórios são produzidos."""

    def analyze(self, dataset: DatasetSnapshot, destination: Path) -> None: ...


class ModelExplanationService(Protocol):
    """Define como explicações e gráficos de um modelo são gerados."""

    def explain(self, artifact: ModelArtifact, dataset: DatasetSnapshot, destination: Path) -> None: ...
