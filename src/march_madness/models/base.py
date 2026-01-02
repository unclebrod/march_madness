"""Base class for numpyro models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import dill as pickle
import optax
from jax import random
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide, initialization
from pydantic import BaseModel

from march_madness.logger import logger
from march_madness.path import OUTPUT_DIR


class SavePayload(BaseModel):
    # TODO: consider expanding with things like number of steps, elbo, summary, acceptance rate, etc.
    samples: dict[str, Any]


class BaseNumpyroModel(ABC):
    name: ClassVar[str]

    def __init__(self):
        self.infer = None
        self.samples = None
        self.results = None

    @abstractmethod
    def model(self, data: Any, **kwargs) -> None: ...

    def fit(
        self,
        data: Any,
        inference: Literal["mcmc", "svi"] = "svi",
        num_warmup: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 4,
        learning_rate: float = 0.01,
        num_steps: int = 25_000,
        seed: int = 0,
        **kwargs,
    ):
        rng_key = random.PRNGKey(seed)
        init_strategy = initialization.init_to_median(num_samples=100)
        match inference:
            case "mcmc":
                kernel = NUTS(self.model, init_strategy=init_strategy)
                self.infer = MCMC(
                    sampler=kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                self.infer.run(rng_key, data)
                self.infer.print_summary()
                self.samples = self.infer.get_samples()

            case "svi":
                schedule = optax.linear_onecycle_schedule(
                    peak_value=learning_rate,
                    transition_steps=num_steps,
                )
                optimizer = optax.adamw(learning_rate=schedule)
                guide = autoguide.AutoNormal(self.model)
                self.infer = SVI(
                    model=self.model,
                    guide=guide,
                    optim=optimizer,
                    loss=Trace_ELBO(),
                    **kwargs,
                )
                self.results = self.infer.run(rng_key, num_steps, data)
                rng_key, rng_key_ = random.split(rng_key)
                self.samples = guide.sample_posterior(
                    rng_key=rng_key_,
                    params=self.results.params,
                    sample_shape=(num_samples,),
                )

            case _:
                raise NotImplementedError("The provided inference method is not implemented.")

    def predict(self, data: Any, seed: int = 0) -> dict[str, Any]:
        if self.samples is None:
            raise ValueError("Model must be fitted before predictions can be made.")
        rng_key = random.PRNGKey(seed)
        predictive = Predictive(self.model, self.samples)
        return predictive(rng_key, data=data)

    def save(self, path: str = "M") -> None:
        save_to = OUTPUT_DIR / f"{path}/{self.name}/model.pkl"
        payload = SavePayload(samples=self.samples)
        with Path(save_to).open("wb") as f:
            pickle.dump(payload.model_dump(), f)
        logger.info(f"Saved model to {save_to}")

    @classmethod
    def load(cls, path: str = "M") -> Self:
        load_from = OUTPUT_DIR / f"{path}/{cls.name}/model.pkl"
        with Path(load_from).open("rb") as f:
            model = pickle.load(f)
        payload = SavePayload.model_validate(model)
        logger.info(f"Loaded model from {load_from}")
        return cls(samples=payload.samples)
