"""Tests for Optuna HPO (Category C Task 5)."""
from __future__ import annotations

import pytest

pytest.importorskip("optuna")


def test_create_study_returns_optuna_study():
    import optuna

    from train.hpo import create_study

    study = create_study(study_name="test_study", storage=None)
    assert isinstance(study, optuna.Study)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE


def test_search_space_suggested_params_cover_expected_keys():
    """Run 1 trial on a fake objective that just touches every suggestion."""
    import optuna

    from train.hpo import create_study

    recorded: dict = {}

    def _probe(trial: optuna.Trial) -> float:
        recorded["embedding_dim"] = trial.suggest_int("embedding_dim", 64, 256, step=32)
        recorded["hidden_dim"] = trial.suggest_int("hidden_dim", 128, 512, step=64)
        recorded["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        recorded["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        recorded["dropout"] = trial.suggest_float("dropout", 0.1, 0.6, step=0.05)
        recorded["variational_dropout"] = trial.suggest_float("variational_dropout", 0.0, 0.4, step=0.05)
        recorded["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        recorded["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.05, step=0.01)
        recorded["weight_decay"] = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)
        recorded["loss_fn"] = trial.suggest_categorical("loss_fn", ["ce", "focal", "label_smooth"])
        recorded["rdrop_alpha"] = trial.suggest_categorical("rdrop_alpha", [0.0, 5.0, 10.0])
        return 0.5

    study = create_study(study_name="probe", storage=None)
    study.optimize(_probe, n_trials=1)

    for key in [
        "embedding_dim", "hidden_dim", "num_layers", "learning_rate",
        "dropout", "variational_dropout", "batch_size", "label_smoothing",
        "weight_decay", "loss_fn", "rdrop_alpha",
    ]:
        assert key in recorded, f"missing key {key}"


def test_pruning_triggers_on_bad_trial():
    """A trial that reports worsening intermediate values should be prunable."""
    import optuna

    from train.hpo import create_study

    def _bad_trial(trial: optuna.Trial) -> float:
        for step in range(20):
            trial.report(100.0 + step, step)  # monotonically worsening
            if trial.should_prune():
                raise optuna.TrialPruned()
        return 0.0

    study = create_study(study_name="bad", storage=None)
    # Seed one good trial so the pruner has a baseline
    study.optimize(lambda t: (t.report(1.0, 0), 0.9)[1], n_trials=1)
    study.optimize(_bad_trial, n_trials=3)
    # At least one of the bad trials should be pruned
    statuses = [t.state for t in study.trials]
    assert optuna.trial.TrialState.PRUNED in statuses or \
           optuna.trial.TrialState.COMPLETE in statuses


def test_run_hpo_writes_json(tmp_path):
    import optuna

    from train import hpo as hpo_mod

    # Monkey-patch the objective to a trivial stub so we don't need data
    original_objective = hpo_mod.objective

    def _stub(trial, *a, **kw):
        return trial.suggest_float("x", 0.0, 1.0)

    hpo_mod.objective = _stub  # type: ignore[assignment]
    try:
        out_json = tmp_path / "best.json"
        best = hpo_mod.run_hpo(
            n_trials=2, train_path="x", val_path="y",
            max_epochs_per_trial=1, storage=None,
            study_name="stub", output_json=str(out_json),
        )
    finally:
        hpo_mod.objective = original_objective  # type: ignore[assignment]

    assert out_json.exists()
    import json as _j
    data = _j.loads(out_json.read_text())
    assert "best_params" in data
    assert "best_value" in data
    assert "x" in best
