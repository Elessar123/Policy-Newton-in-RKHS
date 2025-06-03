import time
import argparse
import numpy as np
import gymnasium as gym
import joblib
import random
import os
import inspect
from datetime import datetime
from distutils.util import strtobool
import shutil

from algo._reinforce_RKHS_Stochastic import RKHSLinearPolySGD
import wandb


def simulate_trajectories(envs, agent, policy, horizon):
    n = envs.num_envs
    observation_dim = envs.single_observation_space.shape
    state_dim = (agent.num_features,)
    action_dim = envs.single_action_space.shape
    num_actions = envs.single_action_space.n  # number of discrete actions

    # Initializing simulation matrices for the given batched episode
    observations = np.zeros((horizon, n) + observation_dim, dtype=np.float32)
    states = np.zeros((horizon, n) + (agent.num_features,), dtype=np.float32)
    actions = np.zeros((horizon, n), dtype=np.int32)
    action_probs = np.zeros((horizon, n, agent.num_actions), dtype=np.float32)
    rewards = np.zeros((horizon, n), dtype=np.float32)
    dones = np.ones((horizon, n), dtype=bool)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    m = horizon  # Default to horizon if no episode ends early

    for t in range(horizon):

        state = agent.get_state(obs)  # (batch_size, num_features)
        action, action_prob = policy(obs)  # (batch_size, ), (batch_size, num_actions)

        observations[t] = obs
        states[t] = state
        actions[t] = action
        action_probs[t] = action_prob
        dones[t] = done

        obs, reward, terminated, truncated, info = envs.step(action)
        done = done | (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`
        reward = reward * ~done
        rewards[t] = reward

        if done.all():
            m = t + 1  # Record the time step where all environments are done
            break

    cum_discounted_rewards = agent.discount_cumsum(rewards[:m], dones[:m], gamma=agent.discount_factor, normalize=False)
    cum_discounted_rewards = np.array(cum_discounted_rewards).astype(np.float32)
    # Avoid division by zero
    sum_not_dones = np.sum(~dones[:m], axis=0)
    sum_not_dones[sum_not_dones == 0] = 1
    mean_episode_return = np.sum(cum_discounted_rewards, axis=0) / sum_not_dones

    traj_info = {
        'observations': observations[:m],
        'states': states[:m],
        'actions': actions[:m],
        'action_probs': action_probs[:m],
        'rewards': rewards[:m],
        'cum_discounted_rewards': cum_discounted_rewards[:m],
        'mean_episode_return': mean_episode_return,
    }

    return traj_info, dones[:m], np.sum(rewards, axis=0), mean_episode_return


def parse_args():
    parser = argparse.ArgumentParser()

    # environment specific args
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
                        help="the id of the gym environment")
    parser.add_argument("--env-seed", type=int, default=1,
                        help="the seed of the gym environment")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed of all rngs")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will save videos")

    # agent specific args
    parser.add_argument("--alpha", type=float, default=1e4,
                        help="the learning rate parameter of the RKHS algorithm")
    parser.add_argument("--normalize-returns", type=lambda x: bool(strtobool(x)), default=True,
                        help="will normalize the returns by standard scaling")
    parser.add_argument("--poly-degree", type=int, default=1,
                        help="the max degree for polynomial features")
    parser.add_argument("--poly-bias", type=lambda x: bool(strtobool(x)), default=False,
                        help="whether to include bias for polynomial features")

    # simulation specific args
    parser.add_argument("--max-timesteps", type=int, default=500,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-updates", type=int, default=1000,
                        help="total update epochs for the policy")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="the number of parallel game environments")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be saved locally")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be tracked on wandb")

    args = parser.parse_args()
    return args


def policy(observation, agent):

    return agent.get_action(observation)


def make_env(gym_id, idx, capture_video, run_name, args):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                b = args.num_updates
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", episode_trigger=lambda x: x % (2 * b // 10) == 0,
                )
        return env
    return thunk


def make_env1(gym_id, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(
            env, f"videos/{run_name}", name_prefix="rl-video-final",
        )
        return env

    return thunk


def get_mp4_files(directory):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


if __name__ == "__main__":
    decay_rate=0.01
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(args.alpha)}__{int(time.time())}"

    if args.track:
        config = vars(args)  # Convert Namespace to dict
        run = wandb.init(
            # Set the project where this run will be logged
            project="rl-crpn-team-research",
            entity='',
            # Track hyperparameters and run metadata
            name=run_name,
            monitor_gym=True,
            config=config,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, i, args.capture_video, run_name, args) for i in range(args.batch_size)],
        copy=False,
    )

    if not args.env_seed == -1:
        envs.reset(seed=args.env_seed)
        envs.action_space.seed(seed=args.env_seed)
        envs.observation_space.seed(seed=args.env_seed)

    if args.capture_video:
        if os.path.exists(f'videos/{args.exp_name}'):
            shutil.rmtree(f'videos/{args.exp_name}')
        os.makedirs(f'videos/{args.exp_name}')

    agent = RKHSLinearPolySGD(
        envs=envs,
        lr=1e-2,
        normalize_returns=args.normalize_returns,
        poly_degree=args.poly_degree,
        set_bias=args.poly_bias,
        kernel='rbf',
        kernel_params={'gamma': 5},
        temperature=100.0
    )

    # Simulation parameters
    max_timesteps = args.max_timesteps
    num_iterations = args.num_updates

    simulation_rewards = []
    simulation_returns = []
    grad_norm_squared = []
    last_video = ''

    for i in range(num_iterations):
        iteration = i
        t_start = time.time()
        traj_info, dones, episodic_rewards, episodic_returns = simulate_trajectories(
            envs, agent, policy=lambda x: policy(x, agent), horizon=max_timesteps
        )
        t1 = time.time() - t_start
        lr_current = agent.lr / (1 + decay_rate * iteration)
        t_start = time.time()
        # curr_grad, curr_Hess, opt_res = agent.learn(traj_info, dones,lr=lr_current)
        curr_grad, curr_Hess, opt_res = agent.learn(traj_info, dones)
        t2 = time.time() - t_start

        simulation_rewards += list(episodic_rewards)
        simulation_returns += list(episodic_returns)
        if curr_grad is not None:
            grad_norm_squared.append(np.linalg.norm(curr_grad) ** 2)
        else:
            grad_norm_squared.append(0.0) 

        avg_traj_length = traj_info['rewards'].shape[0]

        print(f"Iteration {i}, Reward: {np.mean(episodic_rewards)}, "
              f"Delta norm squared: {grad_norm_squared[-1]}, "
              f"T1: {np.round(t1 / avg_traj_length * 1000, 3)}ms, "
              f"T2: {np.round(t2 / avg_traj_length * 1000, 3)}ms")

        if args.track:
            if args.capture_video:
                curr_videos = get_mp4_files(f"videos/{args.exp_name}")
                if len(curr_videos) > 0:
                    curr_video = curr_videos[-1]
                    if curr_video != last_video:
                        v = wandb.Video(
                            data_or_path=curr_video,
                            caption="Training Video",
                            format="mp4"
                        )
                        wandb.log({'videos': v})
                        last_video = curr_video

            for rew, ret in zip(episodic_rewards, episodic_returns):
                wandb.log({'rewards': rew, 'returns': ret})

    print("\nTraining completed.")

    # Close environment
    envs.close()

    # FINAL RUN FOR VIDEO
    if args.capture_video:
        envs1 = gym.vector.SyncVectorEnv(
            [make_env1(args.gym_id, run_name=args.exp_name)]
        )
        traj_info, dones, episodic_rewards, episodic_returns = simulate_trajectories(
            envs1, agent, policy=lambda x: policy(x, agent), horizon=max_timesteps
        )

        final_videos = [i for i in get_mp4_files(f"videos/{args.exp_name}") if 'rl-video-final' in i]
        if final_videos:
            final_video = final_videos[0]
            v = wandb.Video(
                data_or_path=final_video,
                caption="Final Policy Video",
                format="mp4"
            )
            wandb.log({'final-video': v})

        # Close the final environment
        envs1.close()

    if args.save:

        curr_time = datetime.now().strftime('%Y%m%d%H%M%S')

        out_dict = {
            # simulation info
            "episodic_rewards": np.array(simulation_rewards, dtype=np.float32),
            "episodic_returns": np.array(simulation_returns, dtype=np.float32),
            "batch_size": int(args.batch_size),
            "horizon": int(args.max_timesteps),
            "nits": int(args.num_updates),
            "grad_norm_squared": np.array(grad_norm_squared, dtype=np.float32),

            # agent info
            "agent_name": agent.__class__.__name__,
            "trained_agent_params": agent.params,
            "agent_input_keys": list(dict(inspect.signature(agent.__class__).parameters).keys()),

            # env info
            "env_id": str(args.gym_id),
        }
        save_path = f"{os.path.basename(__file__).rstrip('.py')}/{run_name}"
        if not os.path.exists(f'./data/{save_path}'):
            os.makedirs(f'./data/{save_path}')

        for k, v in out_dict.items():
            joblib.dump(v, f"./data/{save_path}/{k}.data")