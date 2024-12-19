"""BNN regression tasks."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import Normal, Uniform


class BNNRegressionProblem(eqx.Module):
    """A non-linear regression problem with a Gaussian likelihood.

    See ``underlying_additive`` for the underlying structure of the data generating
    process.
    """

    dim = 10
    n_test = 1000
    likelihood_std = 3

    def get_data(self, key):
        x_key, y_key = jr.split(key)
        xs = self.get_xs(x_key)
        ys = self.get_ys(y_key, xs)
        return xs | ys

    def get_xs(self, key):
        uniform = Uniform(jnp.full((self.dim,), -4), 4)  # 2D [-4, 4]

        key, subkey = jr.split(key)
        train_x = uniform.sample(subkey, (300,))

        key, subkey = jr.split(key)
        val_x = uniform.sample(subkey, (150,))

        key, subkey = jr.split(key)
        test_x = uniform.sample(subkey, (self.n_test,))

        return {
            "train_x": train_x,
            "val_x": val_x,
            "test_x": test_x,
        }

    def get_ys(self, key, xs):
        return {
            f"{k.split("_")[0]}_y": self.sample_likelihood(seed, x)
            for ((k, x), seed) in zip(xs.items(), jr.split(key, 3), strict=True)
        }

    @eqx.filter_vmap
    def underlying_additive(self, x):
        assert x.shape[0] == self.dim  # Avoid unexpected broadcasting
        return jnp.array(
            [0.1 * x[0] ** 3, jnp.abs(x[1]), -x[2]] + [0 for _ in range(7)],
        )

    def underlying_fn(self, x):
        assert x.ndim == 2
        return jnp.sum(self.underlying_additive(x), axis=1)

    def sample_likelihood(self, key, x):
        return self.underlying_fn(x) + jr.normal(key, x.shape[0]) * self.likelihood_std

    @classmethod
    def log_likelihood(cls, y_hat, y):
        return Normal(scale=cls.likelihood_std).log_prob(y_hat - y).sum()
