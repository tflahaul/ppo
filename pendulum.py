from flax.training.train_state import TrainState
from flax.training import checkpoints
from flax import linen as nn
from typing import NamedTuple

import os
import jax
import jax.numpy as jnp
import fire
import optax
import distrax
import gymnasium as gym


class Transition(NamedTuple):
	done: jnp.ndarray
	action: jnp.ndarray
	value: jnp.ndarray
	reward: jnp.ndarray
	log_prob: jnp.ndarray
	obs: jnp.ndarray


class FFN(nn.Module):
	hidden_dim: int

	@nn.compact
	def __call__(self, inputs: jax.Array) -> jax.Array:
		x = nn.RMSNorm()(inputs)
		ffn_out = nn.relu(nn.Dense(features=self.hidden_dim)(x))
		ffn_out = nn.relu(nn.Dense(features=self.hidden_dim)(ffn_out))
		return x + ffn_out


class ActorCritic(nn.Module):
	num_actions: int
	hidden_dim: int

	@nn.compact
	def __call__(self, inputs: jax.Array) -> jax.Array:
		x = nn.Dense(self.hidden_dim)(inputs)
		x = FFN(self.hidden_dim)(x)
		x = FFN(self.hidden_dim)(x)

		mu     = nn.Dense(features=self.num_actions, name="actor_mu")(x)
		logstd = nn.Dense(features=self.num_actions, name="actor_logstd")(x)

		value = FFN(self.hidden_dim)(x)
		value = FFN(self.hidden_dim)(value)
		value = nn.Dense(features=self.num_actions, name="critic_out")(value)

		return mu, logstd, jnp.squeeze(value, axis=-1)



def normalize(x: jax.Array, eps: float = 1e-8) -> jax.Array:
	return (x - x.mean()) / (eps + x.std())


def train_step(state: TrainState, batch: Transition, returns: jax.Array, eps_clip: float):

	@jax.jit
	def loss_fn(parameters) -> float:
		mu, logstd, value = state.apply_fn(parameters, batch.obs)
		pi = distrax.MultivariateNormalDiag(mu, jnp.exp(logstd))
		log_prob = pi.log_prob(batch.action)

		ratios = jnp.exp(log_prob - batch.log_prob)
		advantages = normalize(returns - batch.value)
		surr1 = ratios * advantages
		surr2 = ratios.clip(1.0 - eps_clip, 1.0 + eps_clip) * advantages
		policy_loss = -jnp.minimum(surr1, surr2).mean() - 0.01 * pi.entropy().mean()

		value_pred_clipped = batch.value + (value - batch.value).clip(-eps_clip, eps_clip)
		value_losses = jnp.square(value - returns)
		value_losses_clipped = jnp.square(value_pred_clipped - returns)
		value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

		return policy_loss + value_loss

	loss, grads = jax.value_and_grad(loss_fn)(state.params)
	new_state = state.apply_gradients(grads=grads)
	return loss, new_state


def collect_trajectory(
	key: jax.random.PRNGKey,
	envs: gym.vector.SyncVectorEnv,
	state: TrainState,
	num_steps: int,
	gamma: float = 0.99,
	gae_lbd: float = 0.95
) -> Transition:
	last_obs, _ = envs.reset()
	transitions = list()
	for _ in range(num_steps):
		_, key = jax.random.split(key)
		mu, logstd, value = state.apply_fn(state.params, last_obs)
		pi = distrax.MultivariateNormalDiag(mu, jnp.exp(logstd))
		action = pi.sample(seed=key)
		obsv, reward, terminated, truncated, _ = envs.step(action)

		transitions.append(Transition(
			jnp.logical_or(terminated, truncated),
			action,
			value,
			reward,
			pi.log_prob(action),
			last_obs
		))

		last_obs = obsv

	transitions = jax.tree.map(lambda *x: jnp.stack(x), *transitions)

	mu, logstd, last_val = state.apply_fn(state.params, last_obs)
	returns = jnp.zeros_like(transitions.value)
	gae = 0
	for t in reversed(range(num_steps)):
		value_tp = last_val if t == num_steps - 1 else transitions.value[t + 1]
		delta = transitions.reward[t] + gamma * value_tp * (1 - transitions.done[t]) - transitions.value[t]
		gae = delta + gamma * gae_lbd * (1 - transitions.done[t]) * gae
		returns = returns.at[t].set(transitions.value[t] + gae)

	return (transitions, returns)


def main(
	env_name: str = "Pendulum-v1",
	num_envs: int = 4,
	seed: int = 6561,
	lr: float = 1e-3,
	update_epochs: int = 4,
	max_grad_norm: float = 0.5,
	ppo_eps_clip: float = 0.2,
	timesteps: int = 500000,
	num_steps: int = 512,
) -> None:
	key = jax.random.PRNGKey(seed)
	envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])

	f = ActorCritic(envs.single_action_space.shape[0], hidden_dim=256)
	parameters = f.init(key, jnp.zeros((1,) + envs.single_observation_space.shape))
	print(f"Number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(parameters))}")

	tx = optax.chain(
		optax.clip_by_global_norm(max_grad_norm),
		optax.adam(lr, eps=1e-5)
	)
	train_state = TrainState.create(
		apply_fn=jax.jit(f.apply),
		params=parameters,
		tx=tx
	)

	manager = checkpoints.AsyncManager(max_workers=2)
	num_iterations = (timesteps // num_steps // num_envs)
	best_avg_reward = -jnp.inf
	for index in range(1, num_iterations + 1):
		running_loss = 0
		_, key = jax.random.split(key)
		batch = collect_trajectory(key, envs, train_state, num_steps)

		permutation = jax.random.permutation(key, num_steps)
		shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
		minibatches = jax.tree.map(lambda x: jnp.reshape(x, [update_epochs, -1] + list(x.shape[1:])), shuffled_batch)

		for step in range(update_epochs):
			transitions = jax.tree.map(lambda x: jnp.take(x, step, axis=0), minibatches[0])
			returns = minibatches[1][step]
			loss, train_state = train_step(train_state, transitions, returns, ppo_eps_clip)
			running_loss = running_loss + loss
		avg_reward = transitions.reward.mean()

		print(f"#{index:02}\tloss: {running_loss:.3f}  avg_reward: {avg_reward:.3f}")

		if avg_reward > best_avg_reward:
			checkpoints.save_checkpoint(os.path.abspath("."), train_state.params, step=index, overwrite=True, async_manager=manager)
			best_avg_reward = avg_reward


if __name__ == "__main__":
	fire.Fire(main)
