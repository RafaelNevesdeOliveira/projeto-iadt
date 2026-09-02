from pathlib import Path

import pandas as pd

from iadt_ml.domain.entities import DatasetSnapshot, Diagnosis

TARGET_COLUMN = "diagnosis"
IGNORED_COLUMNS = frozenset({"id", "Unnamed: 32", "Unnamed:32"})


class CsvDatasetRepository:
    """Lê o CSV público e o converte para a estrutura de domínio do projeto."""

    def __init__(self, source: Path) -> None:
        """Define o caminho do arquivo CSV que será carregado."""
        self._source = source

    def load(self) -> DatasetSnapshot:
        """Valida a coluna-alvo, remove identificadores e retorna features numéricas."""
        dataframe = pd.read_csv(self._source)
        normalized_columns = dataframe.columns.str.strip()
        dataframe.columns = normalized_columns
        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(f"Dataset must contain the '{TARGET_COLUMN}' column.")

        feature_columns = tuple(
            column
            for column in dataframe.columns
            if column not in IGNORED_COLUMNS and column != TARGET_COLUMN
        )
        feature_frame = dataframe.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        diagnosis_series = dataframe.loc[:, TARGET_COLUMN].astype(str).str.strip()
        labels = tuple(Diagnosis(label) for label in diagnosis_series)
        rows = tuple(tuple(float(value) for value in row) for row in feature_frame.to_numpy())
        return DatasetSnapshot(feature_names=feature_columns, rows=rows, labels=labels)
