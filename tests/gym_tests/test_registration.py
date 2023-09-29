# -*- coding: utf-8 -*-
import numpy as np

from typing import ClassVar
from gymnasium import spaces
import gymnasium
from whynot.gym import envs


class ArgumentEnv(gymnasium.Env):
    SIZE: ClassVar[int] = 5

    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        # Gymnasium requires the action and observations spaces be defined
        # even if they're unneeded for this dummy environment.
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.SIZE - 1, shape=(2,), dtype=np.int_),
                "target": spaces.Box(0, self.SIZE - 1, shape=(2,), dtype=np.int_),
            }
        )


gymnasium.register(
    id="test.ArgumentEnv-v0",
    entry_point="test_registration:ArgumentEnv",
    kwargs={
        "arg1": "arg1",
        "arg2": "arg2",
    },
)


def test_make():
    env = gymnasium.make("HIV-v0")
    assert env.spec is not None
    assert env.spec.id == "HIV-v0"
    assert isinstance(env.unwrapped, envs.ODEEnvBuilder)


def test_make_with_kwargs():
    env = gymnasium.make(
        "test.ArgumentEnv-v0", arg2="override_arg2", arg3="override_arg3"
    )
    assert env.spec is not None
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.get_wrapper_attr("arg1") == "arg1"
    assert env.get_wrapper_attr("arg2") == "override_arg2"
    assert env.get_wrapper_attr("arg3") == "override_arg3"


def test_spec():
    spec = envs.spec("HIV-v0")
    assert spec.id == "HIV-v0"


def test_missing_lookup():
    gymnasium.register(id="Test-v1", entry_point="dummy-entry-point")
    try:
        gymnasium.registry["Unknown-v1"]
    except KeyError:
        pass
    else:
        assert False


def test_malformed_lookup():
    try:
        gymnasium.registry["“Breakout-v0”"]
    except KeyError:
        pass
    else:
        assert False


if __name__ == "__main__":
    test_missing_lookup()
