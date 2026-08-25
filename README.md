# grpo-control

Final Reinforcement Learning project repository.

This codebase compares a population-based GRPO family against PPO, SAC, and TD3 on `dm_control/cartpole-swingup-v0`, `dm_control/acrobot-swingup-v0`, and `CarRacing-v3`.

## What matters

- `Simulation.py`: main training and evaluation entry point.
- `Algorithms/AGRPO.py` and `Algorithms/CGRPO.py`: final population-based methods.
- `Algorithms/PPO.py`, `Algorithms/SAC.py`, `Algorithms/TD3.py`: baselines.
- `utils.py`, `logger.py`, `evaluate_runs.py`: support code for environments, logging, and plots.

## Deliverables

- `GRPO_Project.pdf`: final report.
- `Simulation.py`, `utils.py`, `logger.py`, `evaluate_runs.py`, `Algorithms/*.py`: source code.
- `requirements.txt` and `environment.yml`: environment definitions.

## Archived material

- `archive/legacy/`: historical scripts and older AGRPO/CGRPO variants kept for reference only.

## Generated artifacts

These are experiment outputs and should stay out of the main source deliverable:

- `checkpoints/`
- `runs/`
- `plots/`
- `backups/`
- `sduabdullah/`
- `sdumelih/`
- `sdurobin/`
- `run_times.json`

## Reproducibility

`Simulation.py` creates one run folder per env/algorithm/seed, saves `config.json`, copies the source files used for that run, writes metrics to `metrics.csv`, and stores checkpoints under `checkpoints/`.

`evaluate_runs.py` turns saved metrics into plots after training.

## Maintenance notes

- Keep `main` focused on the final runnable pipeline.
- Historical files now live under `archive/legacy/`.
- `Algorithms/` is an explicit Python package for cleaner imports and editor support.