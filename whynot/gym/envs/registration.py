# """Global registry of environments, for consistency with openai gym."""
import gymnasium

__all__ = [
    "make",
    "register",
    "registry",
    "spec",
]

# Keep for consistency with original API
# pylint:disable-msg=invalid-name
# Have a global registry
registry = gymnasium.registry
# registry = EnvRegistry()


# pylint:disable-msg=redefined-builtin
def register(id, **kwargs):
    """Register the environment."""
    return gymnasium.register(id, **kwargs)


def make(id, **kwargs):
    """Build the environment."""
    return gymnasium.make(id, **kwargs)


def spec(id):
    """View the spec for the environment."""
    return gymnasium.spec(id)


warn_once = True
