from pathlib import Path

from iadt_ml.infrastructure.csv_dataset_repository import CsvDatasetRepository
from iadt_ml.infrastructure.exploratory_analysis_service import (
    CLASS_DISTRIBUTION_FILE,
    CORRELATION_FILE,
    CORRELATION_TABLE_FILE,
    DESCRIPTIVE_STATISTICS_FILE,
    DISTRIBUTIONS_FILE,
    SUMMARY_FILE,
    ExploratoryAnalysisService,
)
from iadt_ml.infrastructure.sklearn_model_explanation_service import (
    CONFUSION_MATRIX_FILE,
    FEATURE_IMPORTANCE_CHART_FILE,
    FEATURE_IMPORTANCE_FILE,
    SHAP_SUMMARY_FILE,
    SklearnModelExplanationService,
)
from iadt_ml.infrastructure.sklearn_training_service import SklearnTrainingService

FIXTURE_PATH = Path("tests/fixtures/breast_cancer_sample.csv")


def test_generates_exploratory_analysis_artifacts(tmp_path: Path) -> None:
    dataset = CsvDatasetRepository(FIXTURE_PATH).load()

    ExploratoryAnalysisService().analyze(dataset, tmp_path)

    expected_files = {
        CLASS_DISTRIBUTION_FILE,
        CORRELATION_FILE,
        CORRELATION_TABLE_FILE,
        DESCRIPTIVE_STATISTICS_FILE,
        DISTRIBUTIONS_FILE,
        SUMMARY_FILE,
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})


def test_generates_explanation_artifacts(tmp_path: Path) -> None:
    dataset = CsvDatasetRepository(FIXTURE_PATH).load()
    artifact = SklearnTrainingService().train(dataset)[1]

    SklearnModelExplanationService().explain(artifact, dataset, tmp_path)

    expected_files = {
        CONFUSION_MATRIX_FILE,
        FEATURE_IMPORTANCE_CHART_FILE,
        FEATURE_IMPORTANCE_FILE,
        SHAP_SUMMARY_FILE,
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})
