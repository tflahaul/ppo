import jax

from typing import Tuple
from flax import linen as nn
from jax import numpy as jnp


init_weights = nn.initializers.orthogonal(scale=jnp.sqrt(2))


class ActorCritic(nn.Module):
	num_actions: int
	actor_hidden_dims: Tuple[int]
	critic_hidden_dims: Tuple[int]

	@nn.compact
	def __call__(self, observation: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
		mu = nn.tanh(nn.Dense(self.actor_hidden_dims[0], kernel_init=init_weights)(observation))
		for hidden_dim in self.actor_hidden_dims[1:]:
			mu = nn.tanh(nn.Dense(hidden_dim, kernel_init=init_weights)(mu))
		mu = nn.Dense(self.num_actions, kernel_init=init_weights)(mu)
		logstd = self.param("logstd", nn.initializers.zeros, (self.num_actions,))  # state-independant

		value = nn.tanh(nn.Dense(self.critic_hidden_dims[0], kernel_init=init_weights)(observation))
		for hidden_dim in self.critic_hidden_dims[1:]:
			value = nn.tanh(nn.Dense(hidden_dim, kernel_init=init_weights)(value))
		value = nn.Dense(1, kernel_init=init_weights)(value)
		value = jnp.squeeze(value, axis=-1)

		return mu, logstd, value
