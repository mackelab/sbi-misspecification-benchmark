# SBI Benchmarking Framework

We want to use the top level script run.py as an entry point to launch the entire benchmarking process with a single command. As for now, by calling benchmark_run.py, it performs inference and evaluates it. As we use Hydra Multirun, we can call `postprocess_callback.py` after the evaluation is done. Here we include consolidations and visualisation of the results. 



## Running Benchmarks

###  Basic Usage

To run the benchmark with default settings:

```bash
python -m src.run
```
or 

```bash
python src/run.py
```

This will use `configs/main.yaml` to configure the task, inference method, metric and random seed. and start `benchmark_run.py`for every run. 




## Structure
The benchmark is composed of the following components:

### Entry Point: run.py
This is the main entry point that:

- loads the configuration using Hydra
- calls the benchmarking logic in benchmark_run.py

### Core Logic: benchmark_run.py
This module handles:

- task initialization (e.g., likelihood-misspecified models)
- calling run_inference(...) to generate posteriors and saving them 
- evaluating results with evaluate_inference(...)

### Output structure
the benchmark produces these outputs:

First we get a folder structure containing all the posterior samples `posterior_samples.pt`. 

```bash


outputs/
└── <TaskClassName>/
    └── <Method>/
        └── sims_<NumSimulations>/
            ├── obs_0/
            │   ├── posterior_samples.pt
            │   └── config_used.yaml
            ├── obs_1/
            │   ├── posterior_samples.pt
            │   └── config_used.yaml   
            ...
```
After every benchmark run is done, `postprocess_callback.py`is called, which is responsible for consolidation and visualization. Therefore we get another folder structure containing the results and plots.

```bash
outputs/
└── results/
    ├── plots
    |   └── Plot.png
    ├── results.csv
    └── <TaskClassName>/
        ├── methods_consolidated.csv
        └── <Method>/
            ├── sims_consolidated.csv
            └── sims_<NumSimulations>/
                └──  metrics.csv           

```