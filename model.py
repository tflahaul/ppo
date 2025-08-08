import jax

from typing import Tuple
from flax import linen as nn
from jax import numpy as jnp


nn.linear.default_kernel_init = nn.initializers.orthogonal(jnp.sqrt(2))


class ActorCritic(nn.Module):
	num_actions: int

	@nn.compact
	def __call__(self, observation: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
		bb = nn.selu(nn.Dense(128, name="IO_Dense_0")(observation))
		bb = nn.selu(nn.Dense(64)(bb))
		bb = nn.selu(nn.Dense(32)(bb))
		mu = nn.Dense(self.num_actions, name="IO_Dense_1")(bb)
		logstd = nn.selu(nn.Dense(self.num_actions, name="IO_Dense_2")(bb))

		value = nn.selu(nn.Dense(128, name="IO_Dense_3")(observation))
		value = nn.selu(nn.Dense(64)(value))
		value = nn.selu(nn.Dense(128)(value))
		value = nn.selu(nn.Dense(64)(value))
		value = nn.selu(nn.Dense(32)(value))
		value = nn.Dense(1, name="IO_Dense_4")(value)
		value = jnp.squeeze(value, axis=-1)

		return mu, logstd, value
