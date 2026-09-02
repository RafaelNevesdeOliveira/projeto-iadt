from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Diagnosis(StrEnum):
    """Representa as duas classes de diagnóstico presentes no dataset."""
    BENIGN = "B"
    MALIGNANT = "M"


@dataclass(frozen=True)
class DatasetSnapshot:
    """Armazena features e diagnósticos de uma parte consistente do dataset."""
    feature_names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    labels: tuple[Diagnosis, ...]

    def __post_init__(self) -> None:
        """Garante que o dataset tenha linhas, rótulos e dimensões compatíveis."""
        row_count = len(self.rows)
        if row_count == 0:
            raise ValueError("Dataset must contain at least one row.")
        if row_count != len(self.labels):
            raise ValueError("Rows and labels must have the same length.")
        if any(len(row) != len(self.feature_names) for row in self.rows):
            raise ValueError("Every row must match the feature count.")


@dataclass(frozen=True)
class DatasetPartition:
    """Agrupa os subconjuntos de treino e teste separados de forma estratificada."""
    training: DatasetSnapshot
    testing: DatasetSnapshot


@dataclass(frozen=True)
class EvaluationMetrics:
    """Agrupa as métricas numéricas calculadas para uma avaliação."""
    accuracy: float
    recall: float
    f1_score: float
    roc_auc: float | None = None


@dataclass(frozen=True)
class ConfusionMatrix:
    """Registra os quatro tipos de resultado de uma classificação binária."""
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass(frozen=True)
class ModelEvaluation:
    """Une as métricas de teste à matriz de confusão correspondente."""
    metrics: EvaluationMetrics
    confusion_matrix: ConfusionMatrix


@dataclass(frozen=True)
class ModelArtifact:
    """Representa um pipeline treinado, seus resultados e avaliação final opcional."""
    name: str
    pipeline: Any
    metrics: EvaluationMetrics
    test_evaluation: ModelEvaluation | None = None


@dataclass(frozen=True)
class TrainingReport:
    """Expõe os candidatos treinados e o modelo escolhido pelo caso de uso."""
    artifacts: tuple[ModelArtifact, ...]
    selected_model: ModelArtifact
