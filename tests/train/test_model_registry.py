"""Tests for MLflow Model Registry utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from train.model_registry import (
    DEFAULT_MODEL_NAME,
    get_champion_uri,
    promote_to_champion,
    register_model,
    rollback,
)


class TestRegisterModel:
    """Test model registration."""

    @patch("mlflow.register_model")
    def test_register_model_builds_correct_uri(self, mock_register: MagicMock) -> None:
        """register_model should build runs:/{run_id}/model URI."""
        mock_result = MagicMock()
        mock_result.version = "3"
        mock_register.return_value = mock_result

        version = register_model("abc123", "TestModel")

        mock_register.assert_called_once_with(
            "runs:/abc123/model", "TestModel",
        )
        assert version == "3"

    @patch("mlflow.register_model")
    def test_register_model_default_name(self, mock_register: MagicMock) -> None:
        """register_model should use DEFAULT_MODEL_NAME when no name given."""
        mock_result = MagicMock()
        mock_result.version = "1"
        mock_register.return_value = mock_result

        register_model("run_42")

        mock_register.assert_called_once_with(
            "runs:/run_42/model", DEFAULT_MODEL_NAME,
        )


class TestPromoteToChampion:
    """Test champion alias management."""

    @patch("mlflow.MlflowClient")
    def test_promote_calls_set_alias(self, mock_client_cls: MagicMock) -> None:
        """promote_to_champion should call set_registered_model_alias."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        promote_to_champion("TestModel", "5")

        mock_client.set_registered_model_alias.assert_called_once_with(
            "TestModel", "champion", "5",
        )


class TestGetChampionUri:
    """Test champion URI generation."""

    def test_default_model_name(self) -> None:
        """get_champion_uri with default name."""
        uri = get_champion_uri()
        assert uri == f"models:/{DEFAULT_MODEL_NAME}@champion"

    def test_custom_model_name(self) -> None:
        """get_champion_uri with custom name."""
        uri = get_champion_uri("MyModel")
        assert uri == "models:/MyModel@champion"


class TestRollback:
    """Test rollback (delegates to promote_to_champion)."""

    @patch("mlflow.MlflowClient")
    def test_rollback_sets_champion_to_old_version(self, mock_client_cls: MagicMock) -> None:
        """rollback should reassign champion alias to the given version."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        rollback("TestModel", "2")

        mock_client.set_registered_model_alias.assert_called_once_with(
            "TestModel", "champion", "2",
        )
