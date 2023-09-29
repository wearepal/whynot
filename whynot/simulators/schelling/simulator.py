"""Simulate and analyze runs from the Schelling segregation model."""
import numpy as np
import pandas as pd
import dataclasses

from mesa.batchrunner import batch_run

from whynot.simulators.schelling.model import Schelling


@dataclasses.dataclass
class Config:
    # pylint:disable-msg=too-few-public-methods
    """Parameterization of the Schelling dynamics."""

    #: Vertical size of the grid
    height: int = 10
    #: Horizontal size of the grid
    width: int = 10
    #: Agent density
    density: float = 0.8
    #: Fraction of "minority" agents
    minority_pc: float = 0.2
    #: Agent propensity to live near other agents of the same type.
    homophily: float = 5.0
    #: How much education changes homophily
    education_boost: float = -1.0
    #: What percentage of agent receive education
    education_pc: float = 0.0


def get_segregation(model):
    """Find the fraction of agents that only have neighbors of their same type."""
    segregated_agents = 0
    for agent in model.schedule.agents:
        segregated = True
        for neighbor in model.grid.neighbor_iter(agent.pos):
            if neighbor.type != agent.type:
                segregated = False
                break
        if segregated:
            segregated_agents += 1

    if model.schedule.get_agent_count() == 0:
        return 0.0

    return segregated_agents / model.schedule.get_agent_count()


def simulate(config, rollouts=10, seed=None):
    """Simulate repeated runs of the Schelling model for config.

    Parameters
    ----------
        config: whynot.simulators.Schelling
            Configuration of the grid, agent properties, and model dynamics.
        rollouts: int
            How many times to run the model for the same configuration.
        seed: int
            (Optional) Seed all randomness in rollouts

    Returns
    -------
        segregated_fraction: pd.Series
            What fraction of the agents are segrated at the end of the run
            for each rollout.

    """
    rng = np.random.RandomState(seed)
    parameters = dataclasses.asdict(config)
    parameters |= {"seed": rng.randint(9999999, size=rollouts)}
    model_reporters = {"Segregated_Agents": get_segregation}
    parameters |= {"model_reporters": model_reporters}

    param_sweep = batch_run(
        model_cls=Schelling,
        parameters=parameters,
        max_steps=200,
        display_progress=False,
        number_processes=1,
        data_collection_period=-1,
        # Single rollout for each seed
        iterations=1,
    )
    param_sweep = pd.DataFrame(param_sweep)
    dataframe = param_sweep.get_model_vars_dataframe()
    # Currently just reports the fraction segregated.
    return dataframe.Segregated_Agents.mean()


if __name__ == "__main__":
    print(simulate(Config()))
