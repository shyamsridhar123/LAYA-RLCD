"""Check launcher control flow without downloading models or simulating GPU results."""
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts import setup, train_local


@pytest.fixture
def local_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(train_local, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.json").write_text("[]", encoding="utf-8")
    python = tmp_path / ".venv-train" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    return tmp_path, python


@pytest.mark.parametrize("model,script,final_phase", [
    ("laya", "finetune_colab.py", "finalize"),
    ("decoder", "modernbert_decoder_colab.py", "evaluate"),
])
def test_dispatches_both_phases_with_isolated_data(local_lab, monkeypatch, model, script, final_phase):
    root, python = local_lab
    monkeypatch.setattr(sys, "argv", ["train_local.py", "--model", model])
    run = Mock()
    monkeypatch.setattr(train_local.subprocess, "run", run)
    train_local.main()
    output = root / "runs" / model
    assert (output / "data" / "train.json").read_text() == "[]"
    assert run.call_count == 3  # CUDA check, training, final evaluation.
    for call, phase in zip(run.call_args_list[1:], ("train", final_phase)):
        assert call.args[0] == [str(python), str(root / "scripts" / script), phase, "--root", str(output)]
        assert call.kwargs["cwd"] == root
        assert call.kwargs["check"] is True
        assert call.kwargs["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_existing_run_is_preserved_before_any_subprocess(local_lab, monkeypatch):
    root, _ = local_lab
    output = root / "runs" / "decoder"
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("prior work", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_local.py", "--model", "decoder"])
    run = Mock()
    monkeypatch.setattr(train_local.subprocess, "run", run)
    with pytest.raises(SystemExit, match="already exists"):
        train_local.main()
    run.assert_not_called()
    assert marker.read_text() == "prior work"
    assert list(output.iterdir()) == [marker]


def test_missing_environment_creates_no_run(tmp_path, monkeypatch):
    monkeypatch.setattr(train_local, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_local.py", "--model", "laya"])
    with pytest.raises(SystemExit, match="setup.py --training"):
        train_local.main()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("failure_at", [0, 1])
def test_failed_cuda_or_training_stops_before_evaluation(local_lab, monkeypatch, failure_at):
    root, _ = local_lab
    monkeypatch.setattr(sys, "argv", ["train_local.py", "--model", "decoder", "--run-dir", "runs/custom"])
    run = Mock(side_effect=[None] * failure_at + [subprocess.CalledProcessError(7, "test")])
    monkeypatch.setattr(train_local.subprocess, "run", run)
    with pytest.raises(SystemExit, match="exit 7"):
        train_local.main()
    assert run.call_count == failure_at + 1
    output = root / "runs" / "custom"
    if failure_at == 0:
        assert not output.exists()
    else:
        assert (output / "data" / "train.json").read_text() == "[]"


@pytest.mark.parametrize("option", ["--models", "--download-laya"])
def test_incompatible_setup_flags_do_not_create_environment(tmp_path, monkeypatch, option):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["setup.py", "--training", option])
    create = Mock()
    monkeypatch.setattr(setup.venv, "create", create)
    with pytest.raises(SystemExit) as error:
        setup.main()
    assert error.value.code == 2
    create.assert_not_called()
    assert not list(tmp_path.iterdir())
