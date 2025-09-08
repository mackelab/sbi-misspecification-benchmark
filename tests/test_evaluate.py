import os
from pathlib import Path
import torch
import torch.distributions as D
from src.inference.Run_Inference import run_inference
from src.evaluation.evaluate_inference import evaluate_inference
from Base_Task import BaseTask

class DummyTask(BaseTask):
    def __init__(self, dim=2, noise_std=0.5):
        self.dim = dim
        self.noise_std = noise_std
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )

    def get_simulator(self):
        def sim(theta):
            noise = torch.randn_like(theta) * self.noise_std
            return theta + noise
        return sim

    def get_reference_posterior_samples(self, idx):
        return torch.randn(100, self.dim) * self.noise_std

    def get_observation(self, idx):
        theta = self.prior.sample((1,))
        return theta + torch.randn_like(theta) * self.noise_std

    def get_prior(self):
        return self.prior

    def get_reference_posterior(self, observation):
        mean = torch.zeros(self.dim)
        cov = torch.eye(self.dim)
        return D.MultivariateNormal(mean, cov)


def test_run_inference_and_evaluate(tmp_path):
    os.chdir(tmp_path)

    task = DummyTask()
    method = "NPE"
    metric = "c2st"
    seed = 86
    num_simulations = 100
    num_posterior_samples = 50
    num_observations = 1

    run_inference(
        task,
        method_name=method,
        num_simulations=num_simulations,
        seed=seed,
        num_posterior_samples=num_posterior_samples,
        num_observations=num_observations,
    )

    output_dir = tmp_path / f"outputs/DummyTask_{method}/sims_{num_simulations}/obs_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior_samples_path = output_dir / "posterior_samples.pt"
    torch.save(torch.randn(num_posterior_samples, task.dim), posterior_samples_path)

    x_obs_path = output_dir / "x_obs.pt"
    # Saving observation via torch.save to avoid error
    torch.save(task.get_observation(0), x_obs_path)

    score = evaluate_inference(
        task, method, metric_name=metric, num_simulations=num_simulations
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
