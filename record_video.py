from model import ActorCritic
from flax.training import checkpoints
from functools import partial
from pathlib import Path
from typing import Tuple
import jax
import jax.numpy as jnp
import gymnasium as gym
import fire


def main(
	checkpoint_dir: str,
	video_dir: str = "./agent_videos",
	env_name: str = "Pendulum-v1",
	actor_hidden_dims: Tuple[int] = (256,),
	critic_hidden_dims: Tuple[int] = (128, 64, 128, 64, 128),
	seed: int = 42,
) -> None:
	ckpt_path = Path(checkpoint_dir)
	vid_path = Path(video_dir)

	vid_path.mkdir(parents=True, exist_ok=True)
	print(f"Saving video to: {vid_path.resolve()}")

	env = gym.make(env_name, continuous=True, render_mode="rgb_array")
	env = gym.wrappers.RecordVideo(
		env,
		str(vid_path),
		episode_trigger=lambda ep: ep == 0,
		name_prefix=env_name
	)

	key = jax.random.PRNGKey(seed)
	f = ActorCritic(env.action_space.shape[0], actor_hidden_dims, critic_hidden_dims)

	parameters = f.init(key, jnp.zeros((1,) + env.observation_space.shape))
	parameters = checkpoints.restore_checkpoint(ckpt_path.resolve(), target=parameters)
	infer_fn = jax.jit(partial(f.apply, parameters))

	obs, _ = env.reset()
	terminated = False
	truncated = False
	total_reward = 0
	step_count = 0

	while not (terminated or truncated):
		action, _, _ = infer_fn(obs)
		obs, reward, terminated, truncated, _ = env.step(action)

		total_reward += (reward + 8) / 8
		step_count += 1

	print(f"Episode finished after {step_count} steps.")
	print(f"Total reward: {total_reward:.2f}")

	env.close()


if __name__ == "__main__":
	fire.Fire(main)
