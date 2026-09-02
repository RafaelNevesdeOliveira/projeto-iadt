import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from iadt_ml.domain.entities import DatasetSnapshot

CLASS_DISTRIBUTION_FILE = "class_distribution.png"
CORRELATION_FILE = "correlation_heatmap.png"
CORRELATION_TABLE_FILE = "feature_correlations.csv"
DESCRIPTIVE_STATISTICS_FILE = "descriptive_statistics.csv"
DISTRIBUTIONS_FILE = "feature_distributions.png"
SUMMARY_FILE = "exploratory_summary.json"
FIGURE_WIDTH = 18
FIGURE_HEIGHT = 14
HISTOGRAM_COLUMNS = 5


class ExploratoryAnalysisService:
    """Gera dados e gráficos usados para compreender o dataset antes do treino."""

    def analyze(self, dataset: DatasetSnapshot, destination: Path) -> None:
        """Salva estatísticas, distribuições, correlações e resumo de qualidade."""
        destination.mkdir(parents=True, exist_ok=True)
        dataframe = pd.DataFrame(dataset.rows, columns=dataset.feature_names)
        labels = pd.Series([label.value for label in dataset.labels], name="diagnosis")
        dataframe.to_csv(destination / "processed_dataset.csv", index=False)
        dataframe.describe().transpose().to_csv(destination / DESCRIPTIVE_STATISTICS_FILE)
        correlations = dataframe.corr()
        correlations.to_csv(destination / CORRELATION_TABLE_FILE)
        self._save_summary(dataframe, labels, destination)
        self._save_class_distribution(labels, destination)
        self._save_distributions(dataframe, labels, destination)
        self._save_correlation_heatmap(correlations, destination)

    @staticmethod
    def _save_summary(dataframe: pd.DataFrame, labels: pd.Series, destination: Path) -> None:
        """Registra quantidade de registros, classes, ausências e duplicatas em JSON."""
        class_counts = labels.value_counts().sort_index().to_dict()
        summary = {
            "records": len(dataframe),
            "features": len(dataframe.columns),
            "missing_values": int(dataframe.isna().sum().sum()),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "class_distribution": {label: int(count) for label, count in class_counts.items()},
        }
        (destination / SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _save_class_distribution(labels: pd.Series, destination: Path) -> None:
        """Cria o gráfico com a quantidade de casos benignos e malignos."""
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.countplot(x=labels, order=sorted(labels.unique()), hue=labels, legend=False, ax=axis)
        axis.set_xlabel("Diagnóstico")
        axis.set_ylabel("Quantidade de registros")
        figure.tight_layout()
        figure.savefig(destination / CLASS_DISTRIBUTION_FILE, dpi=180)
        plt.close(figure)

    @staticmethod
    def _save_distributions(
        dataframe: pd.DataFrame,
        labels: pd.Series,
        destination: Path,
    ) -> None:
        """Cria histogramas para comparar cada feature entre os diagnósticos."""
        histogram_rows = int(np.ceil(len(dataframe.columns) / HISTOGRAM_COLUMNS))
        figure, axes = plt.subplots(
            histogram_rows,
            HISTOGRAM_COLUMNS,
            figsize=(FIGURE_WIDTH, histogram_rows * 3),
        )
        for axis, feature_name in zip(axes.flat, dataframe.columns, strict=False):
            sns.histplot(
                data=pd.DataFrame({feature_name: dataframe[feature_name], "diagnosis": labels}),
                x=feature_name,
                hue="diagnosis",
                element="step",
                stat="density",
                common_norm=False,
                ax=axis,
            )
        for axis in axes.flat[len(dataframe.columns) :]:
            axis.remove()
        figure.tight_layout()
        figure.savefig(destination / DISTRIBUTIONS_FILE, dpi=180)
        plt.close(figure)

    @staticmethod
    def _save_correlation_heatmap(correlations: pd.DataFrame, destination: Path) -> None:
        """Cria o mapa de calor que evidencia relações lineares entre features."""
        figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        sns.heatmap(correlations, cmap="coolwarm", center=0, ax=axis)
        figure.tight_layout()
        figure.savefig(destination / CORRELATION_FILE, dpi=180)
        plt.close(figure)
