from pathlib import Path

import joblib

from iadt_ml.domain.entities import ModelArtifact


class JoblibModelRepository:
    """Persiste e recupera pipelines treinados no formato joblib."""

    def save(self, artifact: ModelArtifact, destination: Path) -> Path:
        """Cria diretórios necessários e salva o artefato selecionado."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, destination)
        return destination

    def load(self, source: Path) -> ModelArtifact:
        """Recupera um artefato salvo e confirma que possui o tipo esperado."""
        artifact = joblib.load(source)
        if not isinstance(artifact, ModelArtifact):
            raise TypeError("Saved file does not contain a model artifact.")
        return artifact
