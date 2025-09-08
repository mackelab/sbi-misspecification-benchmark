import os
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from src.utils.benchmark_run import run_benchmark, task_registry
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask

def test_cfg():
    return OmegaConf.create({
        "task": {
            "name": "misspecified_likelihood",
            "dim": 1,
            "tau_m": 1.0,
            "lambda_val": 0.01
        },
        "inference": {
            "method": "npe",
            "num_simulations": 10,
            "num_observations": 2,
            "num_posterior_samples": 5,
        },
        "metric": {"name": "c2st"},
        "random_seed": 42
    })

def test_run_creates_expected_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.chdir(tmp_path)
    # Register the task (run_benchmark needs it)
    task_registry["misspecified_likelihood"] = LikelihoodMisspecifiedTask
    run_benchmark(test_cfg())

    base_dir = tmp_path / "outputs"
    # Look for parameter subfolders under the expected task/method folder to find metrics
    subfolders = list(base_dir.glob("LikelihoodMisspecifiedTask_NPE/*"))
    assert subfolders, "No parameter folders found"
    found_metrics = False
    for param_folder in subfolders:
        for sim_dir in param_folder.glob("sims_*"):
            metrics_file = sim_dir / "metrics.csv"
            if metrics_file.exists():
                found_metrics = True
    assert found_metrics, "No metrics.csv file found in any output simulation folder"

def test_no_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.chdir(tmp_path)
    # Register the task again (for this test)
    task_registry["misspecified_likelihood"] = LikelihoodMisspecifiedTask
    run_benchmark(test_cfg())

    base_dir = tmp_path / "outputs"
    subfolders = list(base_dir.glob("LikelihoodMisspecifiedTask_NPE/*"))
    assert subfolders, "No parameter folders found"
    found_two_rows = False
    for param_folder in subfolders:
        for sim_dir in param_folder.glob("sims_*"):
            metrics_file = sim_dir / "metrics.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                if len(df) == 2:
                    found_two_rows = True
    assert found_two_rows, "No metrics.csv with two rows for two observations found"
