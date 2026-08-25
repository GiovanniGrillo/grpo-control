# grpo-control

Reinforcement Learning project.

This codebase compares a population-based GRPO family against PPO, SAC, and TD3 on `dm_control/cartpole-swingup-v0`, `dm_control/acrobot-swingup-v0`, and `CarRacing-v3`.

## Main files

- `Simulation.py`: training, evaluation, checkpointing, and run logging.
- `Algorithms/AGRPO.py`: main AGRPO implementation.
- `Algorithms/CGRPO.py`: CGRPO population variant.
- `Algorithms/PPO.py`, `Algorithms/SAC.py`, `Algorithms/TD3.py`: baselines.
- `utils.py`: environment wrappers, feature extraction, and buffers.
- `logger.py`: run-folder creation and metric export.
- `evaluate_runs.py`: plotting and comparison utilities.

## Final deliverables

- `GRPO_Project.pdf`: final written report.
- Source code: `Simulation.py`, `utils.py`, `logger.py`, `evaluate_runs.py`, and `Algorithms/*.py`.
- Configuration: `requirements.txt` and `environment.yml`.

## Generated artifacts

These are experiment outputs and stay out of the main source deliverable:

- `checkpoints/`
- `runs/`
- `plots/`
- `backups/`
- `sduabdullah/`
- `sdumelih/`
- `sdurobin/`

## Reproducibility

`Simulation.py` creates one run folder per env/algorithm/seed, saves `config.json`, copies the used source files, writes metrics to `metrics.csv`, and stores checkpoints under `checkpoints/`.

`evaluate_runs.py` reads the saved metrics and generates plots after training.

## Maintenance notes

- Keep `main` focused on the final runnable pipeline.
- Historical files such as `*_old.py` and `*_archive.py` are archival, not the main path.
- `Algorithms/` is an explicit Python package for cleaner imports and editor support.