from flax.training.train_state import TrainState
from flax.training import checkpoints
from typing import NamedTuple, Tuple
from pathlib import Path
from functools import partial
from model import ActorCritic

import jax
import jax.numpy as jnp
import gymnasium as gym
import distrax
import optax
import wandb
import time
import tqdm
import fire


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
	stats = []

	for s in range(rollout_steps):
		_, key = jax.random.split(key)
		mu, logstd, value = state.apply_fn(state.params, last_obs)
		dist = distrax.MultivariateNormalDiag(mu, jnp.exp(logstd))
		action = jnp.clip(dist.sample(seed=key), action_lo, action_hi)
		log_prob = dist.log_prob(action)

		obs, reward, terminated, truncated, infos = envs.step(action)
		buffer.append(Transition(
			jnp.logical_or(terminated, truncated),
			action,
			value,
			reward,
			log_prob,
			last_obs
		))

		last_obs = obs
		if infos:
			stats.append(infos)

	_, _, last_value = state.apply_fn(state.params, last_obs)
	buffer = jax.tree.map(lambda *x: jnp.stack(x), *buffer)
	gae = 0
	returns = []
	for s in reversed(range(rollout_steps)):
		value_tp = last_value if s == rollout_steps - 1 else buffer.value[s + 1]
		mask = 1.0 - buffer.done[s]
		delta = buffer.reward[s] + gamma * value_tp * mask - buffer.value[s]
		gae = delta + gamma * gae_lbd * mask * gae
		returns.insert(0, gae + buffer.value[s])

	return (buffer, jnp.array(returns)), stats


@partial(jax.jit, static_argnums=[3, 4, 5, 6])
def train_step(
	state: TrainState,
	buffer: Transition,
	returns: jax.Array,
	eps_clip: float,
	vf_clip: float,
	vf_coef: float,
	ent_coef: float
) -> Tuple[TrainState, Tuple[float, float, float]]:

	def loss_fn(parameters):
		mu, logstd, value = state.apply_fn(parameters, buffer.obs)
		dist = distrax.MultivariateNormalDiag(mu, jnp.exp(logstd))
		log_prob = dist.log_prob(buffer.action)
		ratios = jnp.exp(log_prob - buffer.log_prob)

		advantages = returns - buffer.value
		advantages = (advantages - advantages.mean()) / (1e-7 + advantages.std())
		surr1 = -advantages * ratios
		surr2 = -advantages * ratios.clip(1 - eps_clip, 1 + eps_clip)
		policy_loss = jnp.mean(jnp.maximum(surr1, surr2))

		value_loss_clipped = buffer.value + jnp.clip(value - buffer.value, -vf_clip, vf_clip)
		value_loss_clipped = jnp.square(value_loss_clipped - returns)
		value_loss = jnp.square(value - returns)
		value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss, value_loss_clipped))

		entropy_loss = jnp.mean(dist.entropy())
		loss = policy_loss - entropy_loss * ent_coef + value_loss * vf_coef
		return loss, (policy_loss, value_loss, entropy_loss)

	grads, losses = jax.grad(loss_fn, has_aux=True)(state.params)
	new_state = state.apply_gradients(grads=grads)
	return new_state, losses


def gen_shuffled_minibatches(key, buffer, minibatch_size, learning_steps):
	buffer = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), buffer)  # flatten
	leaves, treedef = jax.tree.flatten(buffer)
	assert len(leaves[0]) % minibatch_size == 0, "rollout_steps must be divisible by minibatch_size"
	for _ in range(learning_steps):
		key, _sub_key = jax.random.split(key)
		permutation = jax.random.permutation(_sub_key, len(leaves[0]))
		for step in range(len(leaves[0]) // minibatch_size):
			start = step * minibatch_size
			end = start + minibatch_size
			mini_leaves = [leaf[permutation][start:end] for leaf in leaves]
			yield jax.tree.unflatten(treedef, mini_leaves)


def create_env(name: str, reward_clip_range: Tuple[float, float], **kwargs) -> gym.Env:
	env = gym.make(name, **kwargs)
	env = gym.wrappers.RecordEpisodeStatistics(env)
	env = gym.wrappers.NormalizeReward(env)
	env = gym.wrappers.ClipReward(env, *reward_clip_range)
	return env


def main(
	env_name: str = "Pendulum-v1",
	num_envs: int = 20,
	seed: int = 42,
	actor_hidden_dims: Tuple[int] = (256,),
	critic_hidden_dims: Tuple[int] = (128, 64, 128, 64, 128),
	reward_clip_range: Tuple[float, float] = (-10, 10),
	lr: float = 1e-3,
	max_grad_norm: float = 0.5,
	eps_clip: float = 0.2,
	vf_clip: float = 1.0,
	vf_coef: float = 0.5,
	ent_coef: float = 0.01,
	gamma: float = 0.99,
	gae_lbd: float = 0.9,
	timesteps: int = 500000,
	learning_steps: int = 6,
	rollout_steps: int = 512,
	minibatch_size: int = 64,
	save_every: int = 20,
	ckpt_dir: str = "checkpoints/",
) -> None:
	config = locals()
	key = jax.random.PRNGKey(seed)
	envs = gym.vector.SyncVectorEnv([lambda: create_env(env_name, reward_clip_range, continuous=True) for _ in range(num_envs)])

	f = ActorCritic(envs.single_action_space.shape[0], actor_hidden_dims, critic_hidden_dims)
	parameters = f.init(key, jnp.zeros((1,) + envs.single_observation_space.shape))
	print(f"Number of learnable parameters: {sum(x.size for x in jax.tree.leaves(parameters))}")

	tx = optax.chain(
		optax.clip_by_global_norm(max_grad_norm),
		optax.adam(lr, b1=0.9, eps=1e-5)
	)
	train_state = TrainState.create(
		apply_fn=jax.jit(f.apply),
		params=parameters,
		tx=tx
	)
	ckpt_manager = checkpoints.AsyncManager()
	run = wandb.init(project=f"{env_name}-PPO", config=config, name=f"{env_name}_{int(time.time())}")

	for epoch in tqdm.tqdm(range(1, (timesteps // rollout_steps // num_envs) + 1)):
		key, _sub_key = jax.random.split(key)
		buffer, stats = collect_trajectory_segments(_sub_key, envs, train_state, rollout_steps, gamma, gae_lbd)

		running_losses = [0, 0, 0]  # reset losses
		minibatches = gen_shuffled_minibatches(_sub_key, buffer, minibatch_size, learning_steps)
		for transitions, returns in minibatches:
			train_state, (policy_loss, value_loss, entropy_loss) = train_step(
				train_state,
				transitions,
				returns,
				eps_clip,
				vf_clip,
				vf_coef,
				ent_coef
			)
			running_losses[0] = running_losses[0] + policy_loss
			running_losses[1] = running_losses[1] + value_loss
			running_losses[2] = running_losses[2] + entropy_loss

		metrics = {
			"avg_episodic_return": jnp.stack([x["episode"]["r"] for x in stats]).mean() if stats else jnp.nan,
			"losses/policy_loss": running_losses[0],
			"losses/value_loss": running_losses[1],
			"losses/entropy_loss": running_losses[2]
		}
		wandb.log(metrics, step=int(epoch * rollout_steps * num_envs))
		if epoch % save_every == 0:
			checkpoints.save_checkpoint(
				Path(ckpt_dir).resolve(),
				train_state.params,
				step=epoch,
				prefix=f"{env_name}_ckpt",
				keep=3,
				overwrite=True,
				async_manager=ckpt_manager
			)

	ckpt_manager.wait_previous_save()
	run.finish()
	envs.close()


if __name__ == "__main__":
	fire.Fire(main)
