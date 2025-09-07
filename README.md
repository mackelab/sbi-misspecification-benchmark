## Benchmarking Simulation-Based Inference (SBI) for Misspecified Models

## Project Overview

This project aims to develop a benchmark for evaluating Simulation-Based Inference (SBI) methods under model misspecification. 
The benchmark will consist of SBI tasks and model misspecification detection/correction methods to compare their performance and robustness.

## Installation

### Prerequisites
- Python 3.10: Required for compatibility with the SBI package dependencies
- [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html): Recommended package manager to isolate project dependencies.

###  Standard Setup 
**1.) Create and Activate Conda Environment**  
Environment isolation prevents version conflicts between different projects on your system.

```batch
conda create -n sbi-miss-bench python=3.10 -y
```

Switch your terminal session to use the newly created environment.  
You'll need to activate this environment every time you open a new terminal window to work on the project.

```batch
conda activate sbi-miss-bench
```  


**2) Install the sbi package**    
This installs the main SBI package from PyPI along with all necessary dependencies like PyTorch, NumPy, and SciPy. 

```batch
python -m pip install sbi 
```

**3) Clone the repository**  
Get a local copy of the benchmark code and access to all project files.    

```batch
git clone https://github.com/mackelab/sbi-misspecification-benchmark.git
cd sbi-misspecification-benchmark
```

## Background: Simulation-Based Inference (SBI)

Simulation-Based Inference (SBI) uses machine learning methods to estimate parameters for complex scientific models when the likelihood function $p(x|\theta)$ is difficult to calculate. The main goal is to estimate the posterior distribution $p(\theta|x)$, which represents the probability of parameters $\theta$ based on the observed data $x$.

This is typically achieved by:
1.  Defining a simulator model that takes parameters $\theta$ and outputs simulated data $x$.
2.  Specifying a prior distribution for the parameters $\theta$.
3.  Generating many simulations $(\theta, x)$ from the prior and the simulator.
4.  Training a neural network using these simulations to approximate the posterior distribution.
5.  Using the trained network and real-world observations $x_0$ to generate samples from the estimated posterior $p(\theta|x_0)$.

## The Challenge: Model Misspecification

A significant challenge in SBI is *model misspecification*. 
This occurs because the simulator models used are usually simplified and might not accurately reflect the real-world processes generating the observed data $x_0$. 
When the simulator is misspecified, the resulting posterior estimates $p(\theta|x_0)$ can be inaccurate or misleading.

## Project Goal: A Benchmark for Misspecification Methods

The primary goal is to develop a robust benchmark for systematic comparison of SBI methods that address misspecification. This includes:
- Defining a collection of relevant SBI tasks (datasets and simulators).
- Implementing methods to detect/correct misspecification.
- Establishing metrics for comparison (e.g., accuracy, robustness).

## Quickstart
Start the benchmark by running: 
```batch
python src\run.py
```

You can later adjust the parameters in the `main.yaml`configuration file. For this, see [YAML Configuration Guide](docs/YAML_Configuration.md). 

## Expected Outputs

After Hydra multirun jobs complete, the benchmark results are immediately available as `outputs/results/results.csv`and visualized in `outputs/results/plots`.





## Repository & Workflow

This repository serves as the central hub for all code, documentation, issue tracking, and collaboration for this benchmark project.

- **Core Technologies:** Python, Machine Learning Libraries.
- **Version Control:** Git & GitHub. All development discussions, code reviews, and documentation should happen via GitHub Issues and Pull Requests.



    
   
