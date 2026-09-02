import argparse
from pathlib import Path

from iadt_ml.application.use_cases import AnalyzeDataset, ExplainSelectedModel, TrainModels
from iadt_ml.infrastructure.csv_dataset_repository import CsvDatasetRepository
from iadt_ml.infrastructure.exploratory_analysis_service import ExploratoryAnalysisService
from iadt_ml.infrastructure.joblib_model_repository import JoblibModelRepository
from iadt_ml.infrastructure.sklearn_dataset_partitioner import SklearnDatasetPartitioner
from iadt_ml.infrastructure.sklearn_evaluation_service import SklearnEvaluationService
from iadt_ml.infrastructure.sklearn_model_explanation_service import SklearnModelExplanationService
from iadt_ml.infrastructure.sklearn_training_service import SklearnTrainingService

DEFAULT_MODEL_PATH = Path("models/best_model.joblib")
DEFAULT_REPORT_DIRECTORY = Path("reports/figures")


def build_parser() -> argparse.ArgumentParser:
    """Configura os comandos de análise, treino e explicação disponíveis no terminal."""
    parser = argparse.ArgumentParser(prog="iadt-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    analysis_parser = subparsers.add_parser("analyze")
    analysis_parser.add_argument("--dataset", type=Path, required=True)
    analysis_parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("--dataset", type=Path, required=True)
    explain_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    explain_parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    return parser


def main() -> None:
    """Lê o comando informado e monta as dependências para executar cada caso de uso."""
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.command == "train":
        use_case = TrainModels(
            dataset_repository=CsvDatasetRepository(arguments.dataset),
            dataset_partitioner=SklearnDatasetPartitioner(),
            training_service=SklearnTrainingService(),
            evaluation_service=SklearnEvaluationService(),
            model_repository=JoblibModelRepository(),
        )
        report = use_case.execute(arguments.output)
        for artifact in report.artifacts:
            metrics = artifact.metrics
            print(
                f"{artifact.name}: validation_accuracy={metrics.accuracy:.3f}, "
                f"validation_recall={metrics.recall:.3f}, validation_f1={metrics.f1_score:.3f}"
            )
        test_metrics = report.selected_model.test_evaluation.metrics
        print(
            f"test: accuracy={test_metrics.accuracy:.3f}, recall={test_metrics.recall:.3f}, "
            f"f1={test_metrics.f1_score:.3f}, roc_auc={test_metrics.roc_auc:.3f}"
        )
        print(f"selected_model={report.selected_model.name}")
        print(f"model_path={arguments.output}")
    if arguments.command == "analyze":
        AnalyzeDataset(
            dataset_repository=CsvDatasetRepository(arguments.dataset),
            analysis_service=ExploratoryAnalysisService(),
        ).execute(arguments.output_dir)
        print(f"analysis_path={arguments.output_dir}")
    if arguments.command == "explain":
        ExplainSelectedModel(
            dataset_repository=CsvDatasetRepository(arguments.dataset),
            dataset_partitioner=SklearnDatasetPartitioner(),
            model_repository=JoblibModelRepository(),
            explanation_service=SklearnModelExplanationService(),
        ).execute(arguments.model, arguments.output_dir)
        print(f"explanation_path={arguments.output_dir}")
