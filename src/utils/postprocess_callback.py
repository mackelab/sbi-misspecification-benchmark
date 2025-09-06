from pathlib import Path
import pandas as pd

from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from src.utils.LinePlot import LinePlot
from src.utils.consolidate_metrics import consolidate_metrics
from src.tasks.misspecified_tasks import LikelihoodMisspecifiedTask

# Task registry to hold all available task classes
task_registry = {
    "misspecified_likelihood": LikelihoodMisspecifiedTask,
}


class PostProcessCallback(Callback):
    def on_multirun_end(self, config, **kwargs):
        """
        This method is called at the end of a Hydra multirun. It performs the following steps:
        1) Gather run directories from the sweep directory.
        2) Collect run information (task, method, num_simulations, metrics path).
        3) Consolidate metrics and create plots.
        """

        # 1) Gather run directories
        sweep_dir = Path(config.hydra.sweep.dir)                    # Sweep directory of the multirun
        job_dirs = [d for d in sweep_dir.iterdir() if d.is_dir()]   # Job directories of all the single runs

        # 2) Collect run information (config parameters + path to benchmark results) into a DataFrame
        run_records = []    # Each record will hold task, method, num_simulations and the path to its metrics.csv

        for job_dir in job_dirs:
            # Load the config of this job/run
            cfg_path = job_dir / ".hydra" / "config.yaml"
            cfg = OmegaConf.load(cfg_path)

            # Extract relevant config parameters (task, method, num_simulations)
            task_name = config.task.name
            if task_name not in task_registry:
                raise ValueError(f"Unknown task: {task_name}. Available: {list(task_registry.keys())}")

            # Initialize with arbitrary params to only infer the task class name
            task_class_name = task_registry[task_name](1, 1, 1).__class__.__name__

            method = str(cfg.inference.method)
            num_simulations = int(cfg.inference.num_simulations)

            metrics_path = (
                Path("outputs/results")
                / f"{task_class_name}" 
                / f"{method}"
                / f"sims_{num_simulations}"
                / "metrics.csv"
            )

            # Append record
            run_records.append({
                "task": task_class_name,
                "method": method,
                "num_simulations": num_simulations,
                "metrics_path": metrics_path,
            })

        df = pd.DataFrame(run_records)

        # 3) Visualize
        # 3.1) Get the data sources
        metrics_paths = df["metrics_path"].tolist()

        # 3.2) Get the save directory
        # Get unique task-method pairs
        unique_task_methods = df[["task", "method"]].drop_duplicates()

        
        save_directory = Path("outputs/results/plots")

        # 3.3) Consolidate metrics.csv files to metrics_all.csv files within their respective task_method folder
        for _, row in unique_task_methods.iterrows():
            task = row["task"]
            method = row["method"]
            parents_dir = Path("outputs/results") 
            input_dir = parents_dir / f"{task}" / f"{method}" / f"sims_{num_simulations}"
            output_file = parents_dir / f"{task}" / f"{method}" / "sims_consolidated.csv"

            consolidate_metrics(
                input_dir=input_dir,
                output_file=output_file,
                pattern="metrics.csv"
            )

        # 3.4) Create and Save the Plot
        plotter = LinePlot(data_sources=metrics_paths, save_directory=save_directory)
        plotter.run()
