from whynot.gym import envs
from gymnasium.envs.registration import EnvSpec

__all__ = [
    "should_skip_env_spec_for_tests",
    "spec_list",
]


def should_skip_env_spec_for_tests(spec: EnvSpec) -> bool:
    # We skip tests for envs that require dependencies or are otherwise
    # troublesome to run frequently
    spec.entry_point
    return False


spec_list = [
    spec
    for spec in sorted(envs.registry.values(), key=lambda x: x.id)
    if spec.entry_point is not None and not should_skip_env_spec_for_tests(spec)
]
