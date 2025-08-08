from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map
from flax.training import checkpoints
from functools import partial
from pathlib import Path
from typing import NamedTuple, Tuple
from model import ActorCritic

import jax
import jax.numpy as jnp
import gymnasium as gym
import distrax
import optax
import wandb
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
	state: TrainState,
	envs: gym.vector.SyncVectorEnv,
	rollout_steps: int,
	gamma: float,
	gae_lam: float
) -> Tuple[Transition, jax.Array]:
	last_obs, _ = envs.reset()
	buffer = []
	stats = []

	for _ in range(rollout_steps):
		key, _sub_key = jax.random.split(key)
		mu, logstd, value = state.apply_fn(state.params, last_obs)
		dist = distrax.MultivariateNormalDiag(mu, jnp.exp(logstd))
		action = dist.sample(seed=_sub_key)

		obs, reward, terminated, truncated, infos = envs.step(action)
		buffer.append(Transition(
			jnp.logical_or(terminated, truncated),
			action,
			value,
			reward,
			dist.log_prob(action),
			last_obs
		))

		last_obs = obs
		stats.append(infos)

	_, _, last_value = state.apply_fn(state.params, last_obs)
	buffer = jax.tree.map(lambda *x: jnp.stack(x), *buffer)
	returns = jnp.zeros_like(buffer.value)
	mask = 1.0 - buffer.done
	gae = 0
	for s in reversed(range(rollout_steps)):
		value_tp = buffer.value[s + 1] if s != rollout_steps - 1 else last_value
		delta = buffer.reward[s] + gamma * value_tp * mask[s] - buffer.value[s]
		gae = delta + gamma * gae_lam * mask[s] * gae
		returns = returns.at[s].set(gae + buffer.value[s])

	return (buffer, returns), list(filter(bool, stats))


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

		surr = jnp.minimum(advantages * ratios, advantages * jnp.clip(ratios, 1 - eps_clip, 1 + eps_clip))
		clipped = jnp.clip(eps_clip * jnp.exp(-jnp.mean(buffer.log_prob - log_prob)), 0.05, 0.3)
		policy_loss = jnp.where(advantages <= 0, jnp.maximum(surr, advantages * clipped), surr)
		policy_loss = -jnp.mean(policy_loss)

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
	buffer_flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), buffer)
	n_samples = jax.tree.leaves(buffer_flat)[0].shape[0]
	num_minibatches = n_samples // minibatch_size
	assert n_samples % minibatch_size == 0, "rollout_steps must be divisible by minibatch_size"
	for _ in range(learning_steps):
		key, _sub_key = jax.random.split(key)
		perms = jax.random.permutation(_sub_key, n_samples)
		mb = jax.tree.map(lambda x: x[perms].reshape((num_minibatches, minibatch_size) + x.shape[1:]), buffer_flat)
		for idx in range(num_minibatches):
			yield jax.tree.map(lambda x: x[idx], mb)


def create_env(name: str, reward_range: Tuple[float, float], **kwargs):
	env = gym.make(name, **kwargs)
	env = gym.wrappers.RecordEpisodeStatistics(env)
	env = gym.wrappers.TransformReward(env, lambda r: r * 0.1)
	env = gym.wrappers.ClipReward(env, *reward_range)
	return env


def main(
	env_name: str = "LunarLander-v3",
	num_envs: int = 20,
	seed: int = 42,
	reward_range: Tuple[float, float] = (-100, 100),
	lr: float = 1e-3,
	schedule_warmup_steps: int = 200_000,
	schedule_total_steps: int = 2_000_000,
	max_grad_norm: float = 0.5,
	eps_clip: float = 0.2,
	vf_clip: float = 2.0,
	vf_coef: float = 1.0,
	ent_coef: float = 0.01,
	gamma: float = 0.99,
	gae_lam: float = 0.95,
	timesteps: int = 3_000_000,
	learning_steps: int = 4,
	rollout_steps: int = 1024,
	minibatch_size: int = 64,
	save_every: int = 20,
	ckpt_dir: str = "checkpoints/",
) -> None:
	run = wandb.init(project=f"{env_name}-PPO", config=locals())
	envs = gym.vector.SyncVectorEnv([lambda: create_env(env_name, reward_range, continuous=True) for _ in range(num_envs)])
	f = ActorCritic(envs.single_action_space.shape[0])

	key = jax.random.PRNGKey(seed)
	parameters = f.init(key, jnp.zeros((1,) + jnp.shape(envs.single_observation_space)))
	print(f"Number of learnable parameters: {sum(x.size for x in jax.tree.leaves(parameters))}")

	schedule = optax.warmup_cosine_decay_schedule(
		init_value=0,
		peak_value=lr,
		warmup_steps=schedule_warmup_steps // rollout_steps // num_envs * learning_steps,
		decay_steps=(schedule_total_steps - schedule_warmup_steps) // rollout_steps // num_envs * learning_steps,
		end_value=1e-4
	)
	tx = optax.chain(
		optax.clip_by_global_norm(max_grad_norm),
		optax.multi_transform({
			"adam": optax.adamw(schedule, b1=0.9, eps=1e-7),
			"muon": optax.contrib.muon(schedule, eps=1e-7, adam_weight_decay=0.0001)
		}, path_aware_map(lambda path, _: "adam" if any(p.startswith("IO") for p in path) else "muon", parameters))
	)
	train_state = TrainState.create(
		apply_fn=jax.jit(f.apply),
		params=parameters,
		tx=tx
	)
	ckpt_manager = checkpoints.AsyncManager()

	for epoch in tqdm.tqdm(range(1, (timesteps // rollout_steps // num_envs) + 1)):
		key, _sub_key = jax.random.split(key)
		buffer, stats = collect_trajectory_segments(_sub_key, train_state, envs, rollout_steps, gamma, gae_lam)

		running_losses = []  # reset losses
		minibatches = gen_shuffled_minibatches(_sub_key, buffer, minibatch_size, learning_steps)
		for transitions, returns in minibatches:
			train_state, losses = train_step(
				train_state,
				transitions,
				returns,
				eps_clip,
				vf_clip,
				vf_coef,
				ent_coef
			)
			running_losses.append(losses)

		policy_loss, value_loss, entropy_loss = map(lambda x: sum(x) / len(x), zip(*running_losses))
		metrics = {
			"avg_episodic_return": jnp.stack([x["episode"]["r"] for x in stats]).mean() if stats else jnp.nan,
			"losses/policy_loss" : policy_loss,
			"losses/value_loss"  : value_loss,
			"losses/entropy_loss": entropy_loss
		}
		wandb.log(metrics, step=int(epoch * rollout_steps * num_envs), commit=True)
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
