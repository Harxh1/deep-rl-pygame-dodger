import os
import time
import math
import argparse
import numpy as np
from tqdm import tqdm

from envs.dodger_env import DodgerEnv
from dqn.agent import DQNAgent
from utils.plotting import plot_rewards

def train(args):
    env = DodgerEnv(seed=args.seed, render_mode='human' if args.render_every>0 else None, max_steps=args.max_steps)
    obs = env.reset()
    state_dim = obs.shape[0]
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim, lr=args.lr, batch_size=args.batch, buffer_capacity=args.buffer)

    episode_reward = 0.0
    episode = 0
    ep_returns = []
    ep_steps = []
    last_render = 0

    pbar = tqdm(total=args.total_steps, desc='Training', unit='step')
    while agent.total_steps < args.total_steps:
        if args.render_every>0 and (agent.total_steps - last_render) >= args.render_every:
            env.render_mode = 'human'
            last_render = agent.total_steps
        else:
            env.render_mode = None

        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        agent.push(obs, action, reward, next_obs, done)
        agent.total_steps += 1

        loss = agent.train_step()

        episode_reward += reward
        obs = next_obs

        if done:
            ep_returns.append(episode_reward)
            ep_steps.append(agent.total_steps)
            episode += 1
            obs = env.reset()
            episode_reward = 0.0

        pbar.update(1)

    pbar.close()
    env.close()

    os.makedirs('models', exist_ok=True)
    agent.save(os.path.join('models', 'dqn_dodger.pt'))

    os.makedirs('runs', exist_ok=True)
    plot_rewards(ep_steps, ep_returns, os.path.join('runs', 'reward_curve.png'))

    print('Training finished. Saved model to models/dqn_dodger.pt')
    print('Reward curve: runs/reward_curve.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=100000)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--buffer', type=int, default=100000)
    parser.add_argument('--render-every', type=int, default=0, help='Render every N steps (0=off)')
    args = parser.parse_args()
    train(args)
