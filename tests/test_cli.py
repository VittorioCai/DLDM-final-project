from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_help_runs_without_pythonpath_override():
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Train one Colored MNIST shortcut run" in result.stdout


def test_evaluate_help_runs_without_pythonpath_override():
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Recompute metrics for a saved run" in result.stdout
