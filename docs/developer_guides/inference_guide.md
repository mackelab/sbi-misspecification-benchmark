# Inference Guide and How to Add a New Inference Method

This document explains how to use inference method and how to add new methods.
An inference method specifies how posterior distributions are learned or approximated given simulated data.

## Usage of Inference Methods
Inference methods are specified in the YAML configuration file with the `method` parameter. Example:

```yaml
defaults:
  - task: misspecified_likelihood
  - inference: npe
  - metric: ppc
  - _self_
```

or if you want to use Hydra Seeper:

```yaml
defaults:
  - task: misspecified_likelihood
  - inference: npe
  - metric: ppc
  - _self_

random_seed: 86
postprocess: true

methods_sweeper: "npe, nre"

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      inference: ${methods_sweeper}
      inference.num_simulations: ${inference.num_simulations}
```


Available methods include:
- NPE
- NLE
- NRE


## Implementation of a new Inference Method and Example
Each inference method defines:
- how simulations are generated
- how the training is perfromed
- how the posterior is built and sampled

With the following pattern, new inference algorthms can easily be benchmarked.

**1. Set up a Python file**   
As we work with the SBI tool, in this benchmark project, we import the documentation for the inference methods from SBI. If you want to add a method, you can either look for other methods in the SBI tool or implement a new one in `src\inference\methods`.


**2. Add a new Config File**   
If you add another method (either from SBI or implement it by yourself), add a new .yaml config file `src/configs/inference/mymethod.yaml` that specifies the parameters `method`, `num_simulations`, `num_observations` and `num_posterior_samples` for your new method. 

*Example: mymethod.yaml*   
```yaml
method: mymethod 
num_simulations: 500, 1000 
num_observations: 4 
num_posterior_samples: 100
``` 

This config defines the method name and key parameters.   
For a Hydra Multirun add multiple values (e.g. num_simlations: 500, 1000), for a Singlerun, you only need one value. 

Now you can call the new metric in the main config file `main.yaml`. Remember to specify the sweeper parameters in `main.yaml`, if you decided to add multiple parameters.

*Example: Main.yaml Configuartion*
```yaml
defaults:
  - task: misspecified_likelihood     
  - inference: mymethod      # Call new method here
  - metric: c2st       
  - _self_
  ```
For more information on how to configure the `main.yaml` to run the benchmark take a look at the `docs\YAML_Configuration.md`.

**3. Add Tests**   
Remember to add tests for your new method, so that you can ensure correctness.


## 📈 Expected Behavior
`run_inference` will train the chosen method and save `posterior_samples.pt` and `metric.csv`files containing the posterior samples and the result of one run with the specified task, method and metric.
