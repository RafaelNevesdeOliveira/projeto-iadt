import numpy as np
from sklearn.model_selection import train_test_split

from iadt_ml.domain.entities import DatasetPartition, DatasetSnapshot, Diagnosis

RANDOM_SEED = 42
TEST_SIZE = 0.2


class SklearnDatasetPartitioner:
    """Separa dados de forma estratificada usando as ferramentas do scikit-learn."""

    def split(self, dataset: DatasetSnapshot) -> DatasetPartition:
        """Cria conjuntos de treino e teste com a proporção de classes preservada."""
        row_indexes = np.arange(len(dataset.rows))
        labels = np.asarray([label.value for label in dataset.labels])
        training_indexes, testing_indexes = train_test_split(
            row_indexes,
            test_size=TEST_SIZE,
            stratify=labels,
            random_state=RANDOM_SEED,
        )
        return DatasetPartition(
            training=self._snapshot_from_indexes(dataset, training_indexes),
            testing=self._snapshot_from_indexes(dataset, testing_indexes),
        )

    @staticmethod
    def _snapshot_from_indexes(dataset: DatasetSnapshot, indexes: np.ndarray) -> DatasetSnapshot:
        """Cria um novo snapshot usando apenas os índices selecionados na divisão."""
        rows = tuple(dataset.rows[index] for index in indexes)
        labels = tuple(Diagnosis(dataset.labels[index]) for index in indexes)
        return DatasetSnapshot(feature_names=dataset.feature_names, rows=rows, labels=labels)
