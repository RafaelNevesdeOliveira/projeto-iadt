from dataclasses import replace
from pathlib import Path

from iadt_ml.domain.entities import ModelArtifact, TrainingReport
from iadt_ml.domain.ports import (
    DatasetAnalysisService,
    DatasetPartitioner,
    DatasetRepository,
    ModelEvaluationService,
    ModelExplanationService,
    ModelRepository,
    ModelTrainingService,
)


class TrainModels:
    """Coordena a divisão, seleção, avaliação final e persistência de um modelo."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        dataset_partitioner: DatasetPartitioner,
        training_service: ModelTrainingService,
        evaluation_service: ModelEvaluationService,
        model_repository: ModelRepository,
    ) -> None:
        """Recebe as dependências necessárias para executar o treinamento."""
        self._dataset_repository = dataset_repository
        self._dataset_partitioner = dataset_partitioner
        self._training_service = training_service
        self._evaluation_service = evaluation_service
        self._model_repository = model_repository

    def execute(self, destination: Path) -> TrainingReport:
        """Treina no conjunto de treino, avalia uma vez no teste e salva o vencedor."""
        dataset = self._dataset_repository.load()
        partition = self._dataset_partitioner.split(dataset)
        artifacts = self._training_service.train(partition.training)
        selected_model = self._select_model(artifacts)
        test_evaluation = self._evaluation_service.evaluate(selected_model, partition.testing)
        evaluated_model = replace(selected_model, test_evaluation=test_evaluation)
        self._model_repository.save(evaluated_model, destination)
        return TrainingReport(artifacts=artifacts, selected_model=evaluated_model)

    @staticmethod
    def _select_model(artifacts: tuple[ModelArtifact, ...]) -> ModelArtifact:
        """Seleciona por recall, usando F1 e accuracy como critérios de desempate."""
        if not artifacts:
            raise ValueError("Training must produce at least one model.")
        return max(
            artifacts,
            key=lambda artifact: (
                artifact.metrics.recall,
                artifact.metrics.f1_score,
                artifact.metrics.accuracy,
            ),
        )


class AnalyzeDataset:
    """Coordena a geração dos artefatos de análise exploratória."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        analysis_service: DatasetAnalysisService,
    ) -> None:
        """Recebe a fonte de dados e o serviço responsável pela análise."""
        self._dataset_repository = dataset_repository
        self._analysis_service = analysis_service

    def execute(self, destination: Path) -> None:
        """Carrega o dataset e gera tabelas e gráficos no diretório informado."""
        dataset = self._dataset_repository.load()
        self._analysis_service.analyze(dataset, destination)


class ExplainSelectedModel:
    """Coordena a explicação do modelo persistido usando o teste reproduzível."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        dataset_partitioner: DatasetPartitioner,
        model_repository: ModelRepository,
        explanation_service: ModelExplanationService,
    ) -> None:
        """Recebe dependências para recuperar o modelo e explicar suas previsões."""
        self._dataset_repository = dataset_repository
        self._dataset_partitioner = dataset_partitioner
        self._model_repository = model_repository
        self._explanation_service = explanation_service

    def execute(self, model_path: Path, destination: Path) -> None:
        """Recupera o modelo e produz explicações para o conjunto de teste."""
        dataset = self._dataset_repository.load()
        partition = self._dataset_partitioner.split(dataset)
        artifact = self._model_repository.load(model_path)
        self._explanation_service.explain(artifact, partition.testing, destination)
