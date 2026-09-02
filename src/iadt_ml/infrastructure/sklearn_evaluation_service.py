import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from iadt_ml.domain.entities import (
    ConfusionMatrix,
    DatasetSnapshot,
    Diagnosis,
    EvaluationMetrics,
    ModelArtifact,
    ModelEvaluation,
)

MALIGNANT_LABEL = Diagnosis.MALIGNANT.value
CLASS_LABELS = (Diagnosis.BENIGN.value, Diagnosis.MALIGNANT.value)


class SklearnEvaluationService:
    """Calcula métricas e matriz de confusão para o teste final de um modelo."""

    def evaluate(self, artifact: ModelArtifact, dataset: DatasetSnapshot) -> ModelEvaluation:
        """Executa previsões no teste isolado e transforma resultados em métricas."""
        features = np.asarray(dataset.rows, dtype=float)
        labels = np.asarray([label.value for label in dataset.labels])
        predictions = artifact.pipeline.predict(features)
        probabilities = artifact.pipeline.predict_proba(features)[:, 1]
        matrix = confusion_matrix(labels, predictions, labels=CLASS_LABELS)
        metrics = EvaluationMetrics(
            accuracy=float(accuracy_score(labels, predictions)),
            recall=float(recall_score(labels, predictions, pos_label=MALIGNANT_LABEL)),
            f1_score=float(f1_score(labels, predictions, pos_label=MALIGNANT_LABEL)),
            roc_auc=float(roc_auc_score(labels, probabilities)),
        )
        return ModelEvaluation(
            metrics=metrics,
            confusion_matrix=ConfusionMatrix(
                true_negative=int(matrix[0, 0]),
                false_positive=int(matrix[0, 1]),
                false_negative=int(matrix[1, 0]),
                true_positive=int(matrix[1, 1]),
            ),
        )
