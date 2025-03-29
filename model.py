import jax

from flax.linen.initializers import constant, orthogonal
from flax import linen as nn
from jax import numpy as jnp
from typing import Tuple


class ActorCritic(nn.Module):
	num_actions: int
	hidden_dim: int

	@nn.compact
	def __call__(self, observation: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
		mu = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0))(observation)
		mu = nn.tanh(mu)
		mu = nn.Dense(self.num_actions, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0))(mu)
		logstd = self.param("logstd", nn.initializers.zeros, (self.num_actions,))  # state-independant

		value = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0))(observation)
		value = nn.tanh(value)
		value = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0))(value)
		value = nn.tanh(value)
		value = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0))(value)
		value = jnp.squeeze(value, axis=-1)

		return mu, logstd, value
