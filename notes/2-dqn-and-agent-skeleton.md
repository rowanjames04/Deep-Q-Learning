# Video 2 — DQN model & Agent skeleton

This video introduces the two core pieces of the project: the neural network
that approximates the Q-function, and an `Agent` wrapper around the
FlappyBird environment. The agent doesn't *learn* yet — it still acts
randomly — but the scaffolding is now in place.

## What was implemented

### `dqn.py` — the Deep Q-Network

A PyTorch `nn.Module` that approximates **Q(s, a)**: the expected cumulative
reward of taking action `a` in state `s`.

- **Input** — the environment observation (a state vector). For
  `FlappyBird-v0` with `use_lidar=False` this is 12 dimensions
  (`env.observation_space.shape[0]`).
- **Hidden layer** — a single `nn.Linear(state_dim, hidden_dim)` (default
  `hidden_dim=256`) followed by a ReLU activation. This is the only nonlinearity;
  it's what gives the network the capacity to represent a non-trivial
  value function.
- **Output** — `nn.Linear(hidden_dim, action_dim)` producing one Q-value per
  action (`action_dim = 2`: 0 = do nothing, 1 = flap). No activation on the
  output because Q-values are unbounded real numbers (they can be negative).

The `__main__` block is just a smoke test: feed a random `(10, 12)` tensor
through and print the `(10, 2)` Q-value output.

### `agent.py` — the `Agent` wrapper

The previous top-level script (random-action loop) got wrapped in a class
with a `run(is_training=True, render=False)` method.

Key concepts introduced:

- **Device selection** — `'cuda' if torch.cuda.is_available() else 'cpu'`,
  so the network can run on a GPU when one is available. (Marked as
  "can probably be deleted later" — likely because rendering/CPU-only
  training is fine for this small network.)
- **Environment construction** — `gymnasium.make("FlappyBird-v0", ...)`.
  `render_mode="human"` only when `render=True`, so training runs headless
  and fast.
- **Introspecting the env for dims** —
  `num_states = env.observation_space.shape[0]` and
  `num_actions = env.action_space.n`. The network's input/output sizes are
  derived from the env rather than hardcoded, so it stays correct if the
  observation representation changes (e.g. switching on LIDAR).
- **Instantiating the policy network** —
  `policy_dqn = DQN(num_states, num_actions).to_device(device)`. This is the
  *policy* network — the one we'll later use to pick actions. (A *target*
  network and experience replay are the usual next additions.)

The action-selection loop is **unchanged from video 1**: it still calls
`env.action_space.sample()` for a random action. The network has been created
but isn't consulted yet — wiring observation → network → action is the next
step.

### `requirements.txt`

Added `torch` and `torchvision` (switched the deep-learning stack from
TensorFlow, which was listed in video 1 but never used).

## Concepts to remember going forward

- **Q-learning recap** — Q(s, a) is learned by bootstrapping off the Bellman
  equation: `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]`. With a
  neural net, we regress the network's output toward the target
  `r + γ·max_a' Q(s', a')`.
- **Policy vs. target network** — only `policy_dqn` exists so far. Stabilising
  DQN training usually needs a delayed *target* network whose weights are
  periodically copied from the policy network.
- **Exploration** — right now actions are purely random (a crude form of
  ε-greedy with ε=1). The next step is an ε-greedy policy that consults
  `policy_dqn` and decays ε over time.
- **Experience replay** — not yet present. A replay buffer that stores
  `(s, a, r, s', done)` transitions and samples mini-batches is what turns
  this skeleton into a real DQN trainer.