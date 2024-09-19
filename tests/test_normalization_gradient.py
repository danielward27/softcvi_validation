import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import Normal


def test_normalization_grad():
    # Tests the normalization term gradient equals the average score when p^- = q
    # (This is a sanity check of a proof, not a test of package functionality.)

    variational = Normal(jnp.zeros(2))
    negative = variational  # Negative as same as variational

    theta = jr.normal(jr.key(0), (20, *variational.shape))
    guide_params, guide_static = eqx.partition(variational, eqx.is_inexact_array)

    @jax.grad
    def norm_term(guide_params):
        guide = eqx.combine(guide_params, guide_static)
        return jax.scipy.special.logsumexp(
            guide.log_prob(theta) - negative.log_prob(theta),
        )

    norm_term_grad = norm_term(guide_params)
    norm_term_grad = jax.flatten_util.ravel_pytree(norm_term_grad)[0]

    @jax.grad
    def score_term(guide_params):
        guide = eqx.combine(guide_params, guide_static)
        return jnp.mean(guide.log_prob(theta))

    score_term_grad = score_term(guide_params)
    score_term_grad = jax.flatten_util.ravel_pytree(score_term_grad)[0]
    assert pytest.approx(score_term_grad) == norm_term_grad
