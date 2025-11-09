import argparse
import time
import numpy as np
import torch

from envs.dodger_env import DodgerEnv
from dqn.agent import DQNAgent

def main(args):
    env = DodgerEnv(seed=args.seed, render_mode='human', max_steps=args.max_steps)
    obs = env.reset()
    state_dim = obs.shape[0]
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim)
    agent.load(args.checkpoint, map_location='cpu')

    episodes = args.episodes
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            # Greedy action (no epsilon)
            with torch.no_grad():
                s = torch.from_numpy(obs).float().unsqueeze(0)
                q = agent.q(s)
                action = int(torch.argmax(q, dim=1).item())
            obs, r, done, info = env.step(action)
            total += r
        print(f"Episode {ep+1}: return={total:.1f}")
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/dqn_dodger.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=2000)
    args = parser.parse_args()
    main(args)
