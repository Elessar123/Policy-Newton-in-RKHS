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

from algo._crpn_RKHS_stochastic_onebatch_hybrid import RKHSLinearPolyCRPN  
import wandb


def simulate_trajectories(envs, agent, policy, horizon):
    n = envs.num_envs
    observation_dim = envs.single_observation_space.shape
    action_dim = envs.single_action_space.shape
    num_actions = (envs.single_action_space.n,)  # number of discrete actions


    observations = np.zeros((horizon, n) + observation_dim, dtype=np.float32)
    actions = np.zeros((horizon, n) + action_dim, dtype=np.int32)
    action_probs = np.zeros((horizon, n) + num_actions, dtype=np.float32)
    rewards = np.zeros((horizon, n), dtype=np.float32)
    dones = np.ones((horizon, n), dtype=bool)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool) 
    m = horizon 
    for t in range(horizon):

        action, action_prob = policy(obs)  # (bs, ), (bs, action_dim)

        observations[t] = obs
        actions[t] = action
        action_probs[t] = action_prob
        dones[t] = done

        obs, reward, terminated, truncated, info = envs.step(action)
        done = done | (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`
        reward = reward * ~done
        rewards[t] = reward

        if done.all():
            m = t + 1 
            break

    cum_discounted_rewards = agent.discount_cumsum(rewards[:m], dones[:m], gamma=agent.discount_factor, normalize=False)
    cum_discounted_rewards = np.array(cum_discounted_rewards).astype(np.float32)
    valid_steps = ~dones[:m]
    mean_episode_return = np.sum(cum_discounted_rewards, axis=0) / (np.sum(valid_steps, axis=0) + 1e-8) 

    traj_info = {
        'states': observations[:m],         
        'actions': actions[:m],
        'action_probs': action_probs[:m],
        'rewards': rewards[:m],
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
                        help="whether to capture video")
    
    # agent specific args
    parser.add_argument("--alpha", type=float, default=1e1,
                        help="the regularization parameter of the CRPN algorithm")
    parser.add_argument("--normalize-returns", type=lambda x: bool(strtobool(x)), default=True,
                        help="whether to normalize the returns by standard scaling")
    parser.add_argument("--poly-degree", type=int, default=1,
                        help="the max degree for polynomial features")
    parser.add_argument("--poly-bias", type=lambda x: bool(strtobool(x)), default=False,
                        help="whether to include bias for polynomial features")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for rewards")
    parser.add_argument("--predefined-T-steps", type=int, default=5,
                        help="predefined number of trajectory steps to use for Hessian computation")
    
    # simulation specific args
    parser.add_argument("--max-timesteps", type=int, default=500,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-updates", type=int, default=500,
                        help="total update epochs for the policy")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="the number of parallel game environments")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=True,
                        help="if toggled, this experiment will be saved locally")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True,
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

    args = parse_args()

    Gamma = 5
    initial_lr = 1e-2
    decay = 0.99
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}ClipNew__Gamma{int(Gamma)}__ALPHA{int(args.alpha)}__Initial_lr{initial_lr}__Decay{decay}__{int(time.time())}"
    
    if args.track:
        config = vars(args)  
        run = wandb.init(
            project="rl-crpn-team-research",
            entity='',
            # Track hyperparameters and run metadata
            monitor_gym=True,
            name=run_name,
            config=config,
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
        # Delete video_folder
        video_folder = f'videos/{args.exp_name}'
        if os.path.exists(video_folder):
            shutil.rmtree(video_folder)
        os.makedirs(video_folder)

    agent = RKHSLinearPolyCRPN(
        envs,
        alpha=args.alpha,
        normalize_returns=args.normalize_returns,
        poly_degree=args.poly_degree,
        set_bias=args.poly_bias,
        kernel='rbf',  
        kernel_params={'gamma': Gamma}, 
        predefined_T_steps=args.predefined_T_steps,
        discount_factor=args.gamma,  
        seed=args.seed
    )

    # Simulation parameters
    max_timesteps = args.max_timesteps
    num_iterations = args.num_updates

    simulation_rewards = []
    simulation_returns = []
    delta_norm_squared, grad_norm_squared = [], []
    last_video = ''
    
    for i in range(num_iterations):

        t_ = time.time()
        traj_info, dones, episodic_rewards, episodic_returns = simulate_trajectories(
            envs, agent, policy=lambda x: policy(x, agent), horizon=max_timesteps
        )
        t1 = time.time() - t_

        t_ = time.time()
        lr = initial_lr * np.power(0.99, i)
        curr_grad, curr_Hess, opt_res = agent.learn(traj_info, dones,lr)
        t2 = time.time() - t_

        simulation_rewards += list(episodic_rewards)
        simulation_returns += list(episodic_returns)
        if curr_grad is not None:
            grad_norm_squared.append(np.linalg.norm(curr_grad))
        else:
            grad_norm_squared.append(0.0)
        if opt_res is not None and opt_res.x is not None:
            delta_norm_squared.append(np.linalg.norm(opt_res.x))
        else:
            delta_norm_squared.append(0.0)

        avg_traj_length = dones.shape[0]

        print(f"Iteration {i+1}, Reward: {np.mean(episodic_rewards):.2f}, "
              f"Grad norm squared: {grad_norm_squared[-1]:.4f}, "
              f"Delta norm squared: {delta_norm_squared[-1]:.4f}, "
              f"T1: {np.round(t1 / avg_traj_length * 1000, 3)}ms, "
              f"T2: {np.round(t2 / avg_traj_length * 1000, 3)}ms", end="\r")

        if args.track:
            if args.capture_video:
                videos = get_mp4_files(f"videos/{args.exp_name}")
                if videos:
                    curr_video = videos[-1]
                    if curr_video != last_video:
                        v = wandb.Video(
                            data_or_path=curr_video,
                            caption=f"Iteration {i+1}",
                            format="mp4"
                        )
                        wandb.log({'videos': v})
                        last_video = curr_video

            # SOME ANALYSIS
            if opt_res is not None and opt_res.success:
                sm = np.dot(opt_res.x, curr_grad) / (np.linalg.norm(opt_res.x) * np.linalg.norm(curr_grad) + 1e-8)
                sr = np.linalg.norm(opt_res.x) / (np.linalg.norm(curr_grad) + 1e-8)
            else:
                sm = 0.0
                sr = 0.0

            for rew, ret in zip(episodic_rewards, episodic_returns):
                wandb.log({'rewards': rew, 'returns': ret})
            wandb.log({'grad_norm_squared': np.linalg.norm(curr_grad)**2 if curr_grad is not None else 0.0,
                       'delta_norm_squared': np.linalg.norm(opt_res.x)**2 if opt_res is not None and opt_res.x is not None else 0.0,
                       'similarity': sm,
                       'step_ratio': sr,
                       'convergence': int(opt_res.success) if opt_res is not None else 0})

    print()  


    envs.close()

    # FINAL RUN FOR VIDEO
    if args.capture_video:
        envs1 = gym.vector.SyncVectorEnv(
            [make_env1(args.gym_id, run_name=args.exp_name)]
        )
        traj_info, dones, episodic_rewards, episodic_returns = simulate_trajectories(
            envs1, agent, policy=lambda x: policy(x, agent), horizon=max_timesteps
        )

        final_video = [i for i in get_mp4_files(f"videos/{args.exp_name}") if 'rl-video-final' in i]
        if final_video:
            final_video_path = final_video[0]
            v = wandb.Video(
                data_or_path=final_video_path,
                caption="Final Episode",
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
            "delta_norm_squared": np.array(delta_norm_squared, dtype=np.float32),
            "grad_norm_squared": np.array(grad_norm_squared, dtype=np.float32),

            # agent info
            "agent_name": agent.__class__.__name__,
            "trained_agent_params": agent.params,
            "agent_input_keys": list(dict(inspect.signature(agent.__class__).parameters).keys()),

            # env info
            "env_id": str(args.gym_id),
        }
        # save_path = f"{os.path.basename(__file__).rstrip('.py')}/{'__'.join(run_name.split('__')[:-1])}"
        save_path = f"{os.path.basename(__file__).rstrip('.py')}/{run_name}"
        if not os.path.exists(f'./data/{save_path}'):
            os.makedirs(f'./data/{save_path}')

        for k, v in out_dict.items():
            joblib.dump(v, f"./data/{save_path}/{k}.data")

    if args.track:
        run.finish()
