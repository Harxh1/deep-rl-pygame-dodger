# Deep RL Games from Scratch — Dodger (PyTorch + Pygame)

A minimal, end-to-end Deep Q-Network (DQN) project where an agent learns to dodge falling obstacles in a tiny Pygame world.
This is designed as a clean starter you can extend into more complex games (Flappy-like, platformers, etc.).

## Features
- Compact **vector observations** (no image processing) for fast iteration
- **DQN** with target network, epsilon decay, and experience replay
- Clear **reward shaping** and termination conditions
- **Matplotlib** learning-curve plots and simple evaluation script

## Quick start

```bash
# 1) (Optional) create venv
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) install deps
pip install -r requirements.txt

# 3) train (defaults are sensible for a demo)
python train.py --total-steps 100000 --render-every 0

# 4) evaluate a trained model
python eval.py --checkpoint models/dqn_dodger.pt --episodes 5 --render 1
```

> Tip: Start with 50k–100k steps to verify learning, then scale up for smoother policies.

## Environment summary
- **State (9-dim)**: `[player_x, player_vx, nearest_obs_x, nearest_obs_y, nearest_obs_vy, gap_left, gap_right, time_since_spawn, speed_scale]` (all normalized to 0–1).
- **Actions (3)**: `0=stay`, `1=left`, `2=right`.
- **Reward**: `+1` per time step alive, small `-0.01` for lateral movement, `-1` on collision, `+2` on passing an obstacle.
- **Episode end**: collision or `max_steps` reached.

## Files
- `envs/dodger_env.py` — Pygame environment with Gym-like API (reset/step/render).
- `dqn/agent.py` — MLP DQN, epsilon-greedy, target network, training utilities.
- `dqn/replay_buffer.py` — simple experience replay.
- `train.py` — training loop (logs, plots).
- `eval.py` — run a trained policy visually.
- `utils/plotting.py` — reward curve plotting.

## Extend ideas
- Swap DQN -> **Double DQN**, **Dueling DQN**, or **PPO**.
- Add **partial observability** (limited vision cone).
- Curriculum: spawn faster obstacles over time.
- Multi-obs: include **n** nearest obstacles instead of one.

## License
MIT
