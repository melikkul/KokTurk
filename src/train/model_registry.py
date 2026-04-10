"""MLflow Model Registry utilities.

Wraps common operations: register model, promote to champion,
rollback to previous version.
"""

from __future__ import annotations

DEFAULT_MODEL_NAME = "Morphological_Analyzer"


def register_model(run_id: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Register a trained model from a run to the Model Registry.

    Args:
        run_id: MLflow run ID containing the model artifact.
        model_name: Registry name for the model.

    Returns:
        The registered model version string.
    """
    import mlflow  # noqa: PLC0415

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    return result.version


def promote_to_champion(model_name: str, version: str) -> None:
    """Set the @champion alias on a specific model version.

    Args:
        model_name: Registry name for the model.
        version: Version string to promote.
    """
    from mlflow import MlflowClient  # noqa: PLC0415

    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", version)


def get_champion_uri(model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Get the URI for the current champion model.

    Args:
        model_name: Registry name for the model.

    Returns:
        MLflow model URI pointing to the champion alias.
    """
    return f"models:/{model_name}@champion"


def rollback(model_name: str, to_version: str) -> None:
    """Emergency rollback: reassign @champion to a previous version.

    Args:
        model_name: Registry name for the model.
        to_version: Version string to roll back to.
    """
    promote_to_champion(model_name, to_version)
