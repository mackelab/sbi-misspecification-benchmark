# Adding a New Task
This docuent explains how to use tasks and how to add a new one.    

A task defines a prior distribution, a simulator (the generative model), how to fetch an observation. 

# Usage of Tasks
Tasks are specified in the YAML configuration file with the `task` parameter. Example:

`main.yaml`:
```yaml
defaults:
  - task: misspecified_likelihood
  - inference: npe
  - metric: ppc
  - _self_
```

`misspecifies_likelihood.yaml`:
```yaml
name: misspecified_likelihood
dim : 2
tau_m: 1.000
lambda_val: 0.000
```
*NOTE*: Be aware of this being an exammple for a task. Keys can change depending on the task!

Available tasks include:
- Misspecified Likelihood
- Linear Gaussian


## Implementation and Example
A task defines the generative process from which simulations are drawn.    

With the following pattern, new tasks can easily be implemented. 

**1. Set up a Python file**    
**1.1 Required Methods**    
All tasks must implement BaseTask. For better understanding especially these methods are important:

- *get_prior(self)*   
returns the prior distribution 

- *simulator(self, thetas: torch.Tensor)*   
simulates observations x given parameters `thetas` under specific model.

- *get_observation(self, idx: int)*
returns a observation for a given index

- *get_reference_posterior_samples(idx)*   
provides ground-truth posterior samples if available.




**1.2 Example**    
For better understanding on how to implement the required methods, take a look at the following example `src/tasks/my_task.py` or for a more complex implemention look at `src\tasks\misspecified_tasks.py` or `src\tasks\linear_gaussian_task.py`. This simple example uses a normal prior and simulates data by adding Gaussian noise.

*Example: simple implementation with Gaussian noise*
```python
import torch
import torch.distributions as D

class MyTask():
    def get_prior(self):
        return D.Normal(0, 1)   # Use normal prior

    def simulator(self, theta):
        return theta + torch.randn_like(theta)  # Add noise

    def get_observation(self, idx):
        return torch.tensor([0.0]) 

    def get_reference_posterior_samples(self, idx):
        return torch.randn(100, 1)
```





**2. Add a new Config File**   
If you add another task, you need to add a new .yaml config file `src/configs/task/mytask.yaml` that specifies the parameters needed for the new task. 


*Example: my_task.yaml*
```yaml
name: my_task
mu: 2.0
sigma: 1.0
```

**Note**: Config keys are task-specific. Each task can define its own parameters (e.g., `tau_m`, `lambda_val` for the misspecified task, or `prior_mu`, `prior_sigma` for the provided example with Gaussian noise). Nevertheless the keys must match the arguments expected by the task's `__init__` method.

Now you can call the new task in the main config file `main.yaml`.   

*Example: Main.yaml Configuration*
```yaml
defaults:
  - task: my_task     # Call new task here
  - inference: npe
  - metric: c2st
  - _self_
  ``` 
For more information on how to configure the `main.yaml` to run the benchmark take a look at the `docs\YAML_Configuration.md`.

**3. Add Tests**   
Remember to add tests for your new method, so that you can ensure correctness.


## Expected Behavior
- The runner will recognize the new task via Hydra.
- Posterior samples and metrics will be saved under *outputs/*.