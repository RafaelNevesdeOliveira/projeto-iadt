from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, make_scorer, recall_score

from iadt_ml.domain.entities import DatasetSnapshot, Diagnosis, ModelArtifact

CLASS_LABELS = [Diagnosis.BENIGN.value, Diagnosis.MALIGNANT.value]
MALIGNANT_LABEL = Diagnosis.MALIGNANT.value
RANDOM_SEED = 42
PERMUTATION_REPEATS = 30
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
FEATURE_IMPORTANCE_CHART_FILE = "feature_importance.png"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
SHAP_SUMMARY_FILE = "shap_summary.png"


class SklearnModelExplanationService:
    """Produz gráficos que tornam previsões e variáveis importantes mais compreensíveis."""

    def explain(self, artifact: ModelArtifact, dataset: DatasetSnapshot, destination: Path) -> None:
        """Gera matriz de confusão, importância por permutação e resumo SHAP."""
        destination.mkdir(parents=True, exist_ok=True)
        features = np.asarray(dataset.rows, dtype=float)
        labels = np.asarray([label.value for label in dataset.labels])
        self._save_confusion_matrix(artifact, features, labels, destination)
        self._save_feature_importance(artifact, features, labels, dataset.feature_names, destination)
        self._save_shap_summary(artifact, features, dataset.feature_names, destination)

    @staticmethod
    def _save_confusion_matrix(
        artifact: ModelArtifact,
        features: np.ndarray,
        labels: np.ndarray,
        destination: Path,
    ) -> None:
        """Salva a tabela visual dos acertos e erros do classificador."""
        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            labels,
            artifact.pipeline.predict(features),
            display_labels=CLASS_LABELS,
            ax=axis,
            cmap="Blues",
        )
        figure.tight_layout()
        figure.savefig(destination / CONFUSION_MATRIX_FILE, dpi=180)
        plt.close(figure)

    @staticmethod
    def _save_feature_importance(
        artifact: ModelArtifact,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: tuple[str, ...],
        destination: Path,
    ) -> None:
        """Mede o impacto de embaralhar cada feature no recall do modelo."""
        results = permutation_importance(
            artifact.pipeline,
            features,
            labels,
            scoring=make_scorer(recall_score, pos_label=MALIGNANT_LABEL),
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        importance_table = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": results.importances_mean,
                "importance_std": results.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        importance_table.to_csv(destination / FEATURE_IMPORTANCE_FILE, index=False)
        figure, axis = plt.subplots(figsize=(10, 8))
        axis.barh(importance_table["feature"], importance_table["importance_mean"])
        axis.invert_yaxis()
        axis.set_xlabel("Redução média no recall após permutação")
        figure.tight_layout()
        figure.savefig(destination / FEATURE_IMPORTANCE_CHART_FILE, dpi=180)
        plt.close(figure)

    @staticmethod
    def _save_shap_summary(
        artifact: ModelArtifact,
        features: np.ndarray,
        feature_names: tuple[str, ...],
        destination: Path,
    ) -> None:
        """Gera o gráfico SHAP com a influência global de cada feature."""
        preprocessor = artifact.pipeline[:-1]
        classifier = artifact.pipeline.named_steps["classifier"]
        transformed_features = preprocessor.transform(features)
        explainer = shap.Explainer(classifier, transformed_features)
        explanation = explainer(transformed_features, check_additivity=False)
        shap_values = SklearnModelExplanationService._malignant_values(explanation.values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            transformed_features,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        figure = plt.gcf()
        figure.tight_layout()
        figure.savefig(destination / SHAP_SUMMARY_FILE, dpi=180)
        plt.close(figure)

    @staticmethod
    def _malignant_values(values: np.ndarray) -> np.ndarray:
        """Seleciona os valores SHAP da classe maligna quando há duas classes."""
        if values.ndim == 3:
            return values[:, :, 1]
        return values
