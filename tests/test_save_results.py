import pandas as pd
import pytest
import src.utils.save_results as save_results

def test_save_results_append_header_mismatch(tmp_path):
    results = {"x": 0.5}
    task = "T3"
    method = "M3"
    num_sim = 1
    obs_idx = 1
    base = tmp_path
    # write metadata
    path1 = save_results.save_results(
        results,
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        file_mode="write",
        a=1
    )
    # append with different metadata
    path2 = save_results.save_results(
        {"y": 1.5},
        task=task,
        method=method,
        num_simulations=num_sim,
        observation_idx=obs_idx,
        base_directory=base,
        file_mode="append",
        b=2
    )
    # Both files should exist and be different
    assert path1.exists()
    assert path2.exists()
    assert path1 != path2
    # Check each file: right headers
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    assert "a" in df1.columns
    assert "b" in df2.columns
