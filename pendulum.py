from flax.training.train_state import TrainState
from flax.training import checkpoints
from typing import NamedTuple, Tuple
from pathlib import Path
from functools import partial
from model import ActorCritic

import jax
import jax.numpy as jnp
import fire
import optax
import distrax
import gymnasium as gym


class Metrics(NamedTuple):
	loss: float
	ploss: float
	vloss: float
	eloss: float

	def __str__(self) -> str:
		return f"loss: {self.loss:.3f}, ploss: {self.ploss:.3f}, vloss: {self.vloss:.3f}, ent_loss: {self.eloss:.3f}"


class Transition(NamedTuple):
	done: jnp.ndarray
	action: jnp.ndarray
	value: jnp.ndarray
	reward: jnp.ndarray
	log_prob: jnp.ndarray
	obs: jnp.ndarray


def collect_trajectory_segments(
	key: jax.random.PRNGKey,
	envs: gym.vector.SyncVectorEnv,
	state: TrainState,
	rollout_steps: int,
	gamma: float,
	gae_lbd: float
) -> Tuple[Transition, jax.Array]:
	last_obs, _ = envs.reset()
	action_lo = envs.single_action_space.low
	action_hi = envs.single_action_space.high
	buffer = []

	for _ in range(rollout_steps):
		_, key = jax.random.split(key)
		mu, logstd, value = state.apply_fn(state.params, last_obs)
		dist = distrax.MultivariateNormalDiag(mu, jnp.exp(logstd))
		action = dist.sample(seed=key)
		log_prob = dist.log_prob(action)

		obsv, reward, terminated, truncated, _ = envs.step(action)
		buffer.append(Transition(
			jnp.logical_or(terminated, truncated),
			jnp.clip(action, action_lo, action_hi),
			value,
			(reward + 8) / 8,  # normalize
			log_prob,
			last_obs
		))

		last_obs = obsv

	buffer = jax.tree.map(lambda *x: jnp.stack(x), *buffer)
	_, _, last_value = state.apply_fn(state.params, last_obs)
	returns = []
	gae = 0
	for s in reversed(range(rollout_steps)):
		value_tp = last_value if s == rollout_steps - 1 else buffer.value[s + 1]
		delta = buffer.reward[s] + gamma * value_tp * (1 - buffer.done[s]) - buffer.value[s]
		gae = delta + gamma * gae_lbd * (1 - buffer.done[s]) * gae
		returns.insert(0, gae + buffer.value[s])

	return buffer, jnp.array(returns)


@partial(jax.jit, static_argnums=[3, 4, 5, 6])
def train_step(
	state: TrainState,
	buffer: Transition,
	returns: jax.Array,
	eps_clip: float,
	vf_clip: float,
	vf_coef: float,
	ent_coef: float
) -> Tuple[TrainState, Metrics]:

	def loss_fn(parameters):
		mu, logstd, value = state.apply_fn(parameters, buffer.obs)
		dist = distrax.MultivariateNormalDiag(mu, jnp.exp(logstd))
		log_prob = dist.log_prob(buffer.action)

		ratios = jnp.exp(log_prob - buffer.log_prob)
		advantages = returns - buffer.value
		advantages = (advantages - advantages.mean()) / (1e-8 + advantages.std())
		surr1 = -advantages * ratios
		surr2 = -advantages * ratios.clip(1 - eps_clip, 1 + eps_clip)
		policy_loss = jnp.mean(jnp.maximum(surr1, surr2))

		value_loss_clipped = buffer.value + jnp.clip(value - buffer.value, -vf_clip, vf_clip)
		value_loss_clipped = jnp.square(value_loss_clipped - returns)
		value_loss = jnp.square(value - returns)
		value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss, value_loss_clipped))

		entropy_loss = jnp.mean(dist.entropy())
		loss = policy_loss - entropy_loss * ent_coef + value_loss * vf_coef
		return loss, Metrics(loss, policy_loss, value_loss, entropy_loss)

	grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
	new_state = state.apply_gradients(grads=grads)
	return new_state, metrics


def gen_shuffled_minibatches(key, buffer, minibatch_size, learning_steps):
	buffer = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), buffer)  # flatten
	leaves, treedef = jax.tree.flatten(buffer)
	assert len(leaves[0]) % minibatch_size == 0, "rollout_step must be divisible by minibatch_size"
	for _ in range(learning_steps):
		_, key = jax.random.split(key)
		permutation = jax.random.permutation(key, len(leaves[0]))
		for step in range(len(leaves[0]) // minibatch_size):
			start = step * minibatch_size
			end = start + minibatch_size
			mb_leaves = [leaf[permutation][start:end] for leaf in leaves]
			yield jax.tree.unflatten(treedef, mb_leaves)


def main(
	env_name: str = "Pendulum-v1",
	num_envs: int = 20,
	seed: int = 42,
	lr: float = 2e-4,
	max_grad_norm: float = 0.5,
	ppo_eps_clip: float = 0.2,
	ppo_vf_clip: float = 10.0,
	vf_coef: float = 0.5,
	ent_coef: float = 0.001,
	gamma: float = 0.99,
	gae_lbd: float = 0.1,
	timesteps: int = 410000,
	learning_steps: int = 6,
	rollout_steps: int = 512,
	minibatch_size: int = 64,
	save_every: int = 20,
) -> None:
	for name, value in locals().items():
		print(f"{name}: {value}")

	key = jax.random.PRNGKey(seed)
	envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])

	f = ActorCritic(envs.single_action_space.shape[0], hidden_dim=128)
	parameters = f.init(key, jnp.zeros((1,) + envs.single_observation_space.shape))
	print(f"Number of learnable parameters: {sum(x.size for x in jax.tree.leaves(parameters))}")

	tx = optax.chain(
		optax.clip_by_global_norm(max_grad_norm),
		optax.lion(lr, b1=0.9)
	)
	train_state = TrainState.create(
		apply_fn=jax.jit(f.apply),
		params=parameters,
		tx=tx
	)
	manager = checkpoints.AsyncManager()

	for epoch in range(1, (timesteps // rollout_steps // num_envs) + 1):
		key, _sub_key = jax.random.split(key)
		buffer = collect_trajectory_segments(_sub_key, envs, train_state, rollout_steps, gamma, gae_lbd)

		running_metrics = Metrics(0, 0, 0, 0)  # reset metrics
		minibatches = gen_shuffled_minibatches(_sub_key, buffer, minibatch_size, learning_steps)
		for mb_transitions, mb_returns in minibatches:
			train_state, metrics = train_step(
				train_state,
				mb_transitions,
				mb_returns,
				ppo_eps_clip,
				ppo_vf_clip,
				vf_coef,
				ent_coef
			)
			running_metrics = jax.tree.map(lambda x, y: x + float(y), running_metrics, metrics)

		print(f"#{epoch:02}\t{buffer[0].reward.mean():.3f} {running_metrics}")

		if epoch % save_every == 0:
			checkpoints.save_checkpoint(Path().resolve(), train_state.params, step=epoch, overwrite=True, async_manager=manager)


if __name__ == "__main__":
	fire.Fire(main)
