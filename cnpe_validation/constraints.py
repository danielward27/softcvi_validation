"""A small set of contraints for transforming defining a mapping to unbounded domain."""

import equinox as eqx
from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Exp,
    Identity,
    Invert,
    Tanh,
)
from jax import Array
from jax.numpy import vectorize


class AbstractConstraint(eqx.Module):
    """Abstract class representing a contraint for a scalar variable."""

    bijection: eqx.AbstractVar[AbstractBijection]

    def __call__(self, x: Array):
        """Convenience vectorized transform to unbounded."""
        return vectorize(self.bijection.transform)(x)


class Real(AbstractConstraint):
    """Represents real/unbounded variables."""

    bijection = Identity()


class Positive(AbstractConstraint):
    """maps positive -> unbounded using log."""

    bijection = Invert(Exp())


class Interval(AbstractConstraint):
    """maps interval -> unbounded by transforming to [-1, 1], and applying arctanh."""

    bijection: Chain

    def __init__(self, low: float, high: float):
        loc = -(low + high) / 2
        scale = 2 / (high - low)
        aff = Affine(loc * scale, scale)
        self.bijection = Chain([aff, Invert(Tanh())])
