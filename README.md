# Deep Q-Learning

A PyTorch implementation of Deep Q-Network (DQN) with optional **Double DQN** and **Dueling DQN** extensions, training on CartPole and FlappyBird via Gymnasium.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python agent.py cartpole1 --train      # CartPole-v1
python agent.py flappybird1 --train    # FlappyBird-v0
```

Training runs indefinitely — stop it manually (`Ctrl-C`) once you're happy with the reward. A VS Code launch config for both environments is in `.vscode/launch.json`.

## Test / watch

```bash
python agent.py cartpole1              # loads runs/cartpole1.pt and renders
```

## Outputs

All run artifacts are written to `runs/` (gitignored):

| File | Contents |
|------|----------|
| `runs/<set>.pt`  | Best-policy model checkpoint (saved on each new best reward) |
| `runs/<set>.log` | Timestamped best-reward log |
| `runs/<set>.png` | Mean reward + epsilon decay plot (refreshed every 10s) |

## Hyperparameters

Each named set in `hyperparameters.yml` configures one run. Notable keys:

- `enable_double_dqn` / `enable_dueling_dqn` — toggle the two DQN extensions.
- `fc1_nodes` — width of the shared hidden layer.
- `dueling_hidden_dim` — width of the dueling value/advantage streams (only used when `enable_dueling_dqn: true`).
- `network_sync_rate` — steps between target-network syncs.
- `epsilon_init` / `epsilon_decay` / `epsilon_min` — exploration schedule (epsilon decays per optimisation step).
- `device` — `'auto'` (CUDA if available else CPU), or `'cpu'` / `'cuda'` to force.
- `seed` — optional integer to seed `env`, `random`, `numpy`, and `torch` for reproducible runs.
- `env_make_params` — optional env-specific kwargs (e.g. `use_lidar: False` for FlappyBird).

## Layout

- `agent.py` — training/evaluation loop, epsilon-greedy action selection, optimisation.
- `dqn.py` — the Q-network (vanilla, dueling, or double+dueling).
- `experience_replay.py` — the replay memory buffer.
- `hyperparameters.yml` — per-run configuration.