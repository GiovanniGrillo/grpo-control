# grpo-control

Final repository for the DM887 Reinforcement Learning project.

The codebase compares a population-based GRPO family against PPO, SAC, and TD3 on continuous-control tasks such as `dm_control/cartpole-swingup-v0`, `dm_control/acrobot-swingup-v0`, and `CarRacing-v3`.

## Main entry points

- `Simulation.py`: training, evaluation, checkpointing, and run logging.
- `Algorithms/AGRPO.py`: the main AGRPO implementation used in the final branch.
- `Algorithms/CGRPO.py`: the CGRPO variant used for population-based updates.
- `Algorithms/PPO.py`, `Algorithms/SAC.py`, `Algorithms/TD3.py`: baseline agents.
- `utils.py`: environment wrappers, feature extraction, and trajectory buffers.
- `logger.py`: run-folder creation, config backup, metric export, and runtime logging.
- `evaluate_runs.py`: plotting and comparison utilities for completed runs.

## Repository taxonomy

- Source: `Simulation.py`, `utils.py`, `logger.py`, `evaluate_runs.py`, and `Algorithms/*.py`.
- Configuration: `requirements.txt` and `environment.yml`.
- Documentation: `README.md` and `CGRPO.pdf`.
- Generated artifacts: `checkpoints/`, `runs/`, `plots/`, `backups/`, `sduabdullah/`, `sdumelih/`, `sdurobin/`.

## Reproducibility workflow

Training is launched from `Simulation.py`. For each environment and algorithm, the script

1. creates a run folder under `runs/`,
2. stores `config.json`,
3. copies the source files used for that run into `runs/<run>/source_code/`,
4. trains each seed,
5. saves checkpoints under `checkpoints/`, and
6. exports flat metrics to `metrics.csv` plus a `run_time.txt` summary.

Evaluation plots can be generated from saved metrics with `evaluate_runs.py`.

## Final deliverables

- `CGRPO.pdf`: final written report.
- Source code in the root Python files and `Algorithms/`.
- Dependency definitions in `requirements.txt` and `environment.yml`.

## Generated artifacts

These folders/files are produced by experiments and are intentionally excluded from the main source deliverable:

- `checkpoints/`
- `runs/`
- `plots/`
- `backups/`
- `sduabdullah/`, `sdumelih/`, `sdurobin/`

## Notes for maintainers

- Keep `main` focused on the final runnable pipeline.
- Historical variants such as `*_old.py`, `*_archive.py`, and similar experimental files should remain separate from the primary training path.
- `Algorithms/` is now an explicit Python package so editor and import resolution stay consistent.