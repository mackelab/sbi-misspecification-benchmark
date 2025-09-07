# Adding a New Metric
This docuent explains how to use metrics and how to add a new one.    

A metric evaluates how close inferred posteriors are to ground-truth reference distributions.


## Usage of Metrics
Metrics are specified in the YAML configuration file with the `metric` parameter. Example:

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

metric_sweeper: "ppc, c2st"

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      metric: ${metric_sweeper}
      inference.num_simulations: ${inference.num_simulations}
```


Available metrics include:
- ppc
- c2st

## Implementation and Example
**1. Set up a Python file**    
If you want to add a new metric, create a new python file `src/evaluation/metrics/mymetric.py`.


**1.1 Required Methods**     
Since metrics can differ significantly in their logic (e.g. C2ST trains a classifier, while PPC generates data and compares it to observations), there is no shared interface that all metrics must implement.       
However, each metric should implement the main function named `compute_mymetric(...){...}` that takes the necessary inputs (e.g. posterior samples, reference data) and returns a scalar score.



**1.2 Example**     
For a better understanding of the main function, take a look at the following short example. This simple metric measures the absolute difference in means between posterior and reference distributions.   

*Example: compute_mymetric*
```python
def compute_mymetric(posterior_samples, reference_samples):
    return float(abs(posterior_samples.mean() - reference_samples.mean()))
```


**2. Add a new Config File**     
If you add another metric, you have to add a new .yaml config file `src/configs/metric/mymetric.yaml`, that follows the simple structure:

*Example: Config File*    
```yaml
name: mymetric
```

Now you can call the new metric in the main config file `main.yaml`.   

*Example: Main.yaml Configuration*
```yaml
defaults:
  - task: misspecified_likelihood     
  - inference: npe
  - metric: mymetric       # Call new metric here
  - _self_
  ```

For a Hydra Multirun add multiple metrics, for a Singlerun, you only need one metric. 

Now you can call the new metric in the main config file `main.yaml`. Remember to specify the sweeper parameters in `main.yaml`, if you decided to add multiple metrics.

For more information on how to configure the `main.yaml` to run the benchmark take a look at the `docs\YAML_Configuration.md`.
 
**3. Add Tests**          
Remember to add tests for your new method, so that you can ensure correctness.


## Expected Behavior
- The runner will recognize the new metric via Hydra.
- Results are appended to metrics.csv for each run.

## Interpretation of Scores

- **~0.5** → The classifier cann't distinguish between inference and reference samples → no or low misspecification   
- **>0.5** → Classifier can distinguish → larger misspecification between inference and reference  
- **Close to 1.0** → Samples are clearly different → inference didn't generate the true posterior  

*NOTE: As the visualization only shows values in [0.5, 1] try to normalize the results you receive from metrics to fit this intervall.* 
