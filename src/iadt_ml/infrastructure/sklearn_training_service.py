import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iadt_ml.domain.entities import DatasetSnapshot, EvaluationMetrics, ModelArtifact

RANDOM_SEED = 42
CROSS_VALIDATION_FOLDS = 3
SCORING = {
    "accuracy": "accuracy",
    "recall": make_scorer(recall_score, pos_label="M"),
    "f1": make_scorer(f1_score, pos_label="M"),
}
LOGISTIC_REGRESSION_NAME = "logistic_regression"
RANDOM_FOREST_NAME = "random_forest"
LOGISTIC_REGRESSION_MAX_ITERATIONS = 2_000
RANDOM_FOREST_ESTIMATORS = 400


class SklearnTrainingService:
    """Treina e compara pipelines de classificação com validação cruzada."""

    def train(self, dataset: DatasetSnapshot) -> tuple[ModelArtifact, ...]:
        """Calcula métricas de validação e ajusta cada candidato em todo o treino."""
        feature_matrix = np.asarray(dataset.rows, dtype=float)
        target_vector = np.asarray([label.value for label in dataset.labels])
        candidates = self._candidates()
        artifacts = tuple(
            self._fit_and_validate(
                name,
                pipeline,
                feature_matrix,
                target_vector,
            )
            for name, pipeline in candidates
        )
        return artifacts

    @staticmethod
    def _candidates() -> tuple[tuple[str, Pipeline], ...]:
        """Monta os pipelines de Regressão Logística e Random Forest."""
        logistic_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=LOGISTIC_REGRESSION_MAX_ITERATIONS,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        random_forest_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=RANDOM_FOREST_ESTIMATORS,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                )
            ]
        )
        return (
            (LOGISTIC_REGRESSION_NAME, logistic_pipeline),
            (RANDOM_FOREST_NAME, random_forest_pipeline),
        )

    @staticmethod
    def _fit_and_validate(
        name: str,
        pipeline: Pipeline,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> ModelArtifact:
        """Valida um pipeline em três dobras e o ajusta nos dados de treino completos."""
        cross_validator = StratifiedKFold(
            n_splits=CROSS_VALIDATION_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        scores = cross_validate(
            pipeline,
            features,
            labels,
            cv=cross_validator,
            scoring=SCORING,
            n_jobs=1,
        )
        pipeline.fit(features, labels)
        metrics = EvaluationMetrics(
            accuracy=float(np.mean(scores["test_accuracy"])),
            recall=float(np.mean(scores["test_recall"])),
            f1_score=float(np.mean(scores["test_f1"])),
        )
        return ModelArtifact(name=name, pipeline=pipeline, metrics=metrics)
