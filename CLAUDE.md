# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

A Deep Q-Network (DQN) learning to play FlappyBird, built by following a
video tutorial series. The codebase is intentionally incremental — each
video adds the next piece, so the repo is usually in a half-finished state
relative to a "complete" DQN.

- **Env:** `flappy-bird-gymnasium` (`gymnasium.make("FlappyBird-v0", ...)`).
  Observation is a 12-dim state vector with `use_lidar=False`. Actions are
  discrete: `0` = do nothing, `1` = flap.
- **DL stack:** PyTorch (`torch`, `torchvision`). TensorFlow is listed in
  `requirements.txt` as a leftover from video 1 but is not used.
- **Key files:**
  - `dqn.py` — the Q-network (`nn.Module`): one hidden Linear layer (256
    units) + ReLU, output is raw Q-values per action.
  - `agent.py` — `Agent` class with `run(is_training=True, render=False)`
    wrapping the env. Currently still selects random actions; the policy
    network is instantiated but not yet consulted.

## Tutorial / notes workflow

- There is a `notes/` folder. **After each video**, write a note summarising
  the concepts just implemented. Name the file `<video-number>-<slug>.md`
  (e.g. `2-dqn-and-agent-skeleton.md`). The note should explain *concepts*,
  not just restate the code.
- **What counts as "this video's work":** the uncommitted / untracked changes
  on top of the last commit are the latest video's implementation. The
  committed state is everything from prior videos.
- After the note (and any other requested files) are written, commit with a
  **conventional-commit** one-liner (e.g. `docs: add video 2 notes on DQN &
  agent skeleton`).

## Conventions

- Keep the incremental, tutorial-following style — don't refactor ahead of
  where the tutorial is.
- Comments are sparse and explanatory (see `agent.py`). Match that density.
- `device = 'cuda' if torch.cuda.is_available() else 'cpu'` is flagged in
  `agent.py` as "can probably be deleted later" — leave it unless the
  tutorial removes it.