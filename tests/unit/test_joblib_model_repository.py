from pathlib import Path

import joblib

from iadt_ml.domain.entities import EvaluationMetrics, ModelArtifact
from iadt_ml.infrastructure.joblib_model_repository import JoblibModelRepository


def test_saves_artifact_and_creates_parent_directories(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "model.joblib"
    artifact = ModelArtifact(
        name="random_forest",
        pipeline={"ready": True},
        metrics=EvaluationMetrics(accuracy=0.9, recall=0.8, f1_score=0.85),
    )

    saved_path = JoblibModelRepository().save(artifact, destination)

    assert saved_path == destination
    assert destination.exists()
    loaded = joblib.load(destination)
    assert loaded.name == "random_forest"
    assert loaded.metrics.recall == 0.8
    assert JoblibModelRepository().load(destination).name == "random_forest"
