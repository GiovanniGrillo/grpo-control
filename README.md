# grpo-control

Reinforcement learning project focused on comparing population-based GRPO variants against PPO, SAC, and TD3 on continuous control and racing tasks.

This project covers two main families of environments:
- `dm_control/cartpole-swingup-v0` and `dm_control/acrobot-swingup-v0`: classical continuous-control benchmarks.
- `CarRacing-v3`: a driving/racing environment requiring stable control and long-horizon decision making.

## Tech Stack

<div align="left">
  <img src="https://img.shields.io/badge/Python-3.12.9-3776AB?logo=python&logoColor=white" alt="Python 3.12.9" />
  <img src="https://img.shields.io/badge/PyTorch-2.11.0%2Bcu130-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.11.0+cu130" />
  <img src="https://img.shields.io/badge/Gymnasium-1.2.3-4B8BBE?logo=openai&logoColor=white" alt="Gymnasium 1.2.3" />
  <img src="https://img.shields.io/badge/dm_control-1.0.40-4E9A06?logo=python&logoColor=white" alt="dm_control 1.0.40" />
  <img src="https://img.shields.io/badge/JAX-0.9.2-FF6F00?logo=python&logoColor=white" alt="JAX 0.9.2" />
  <img src="https://img.shields.io/badge/NumPy-2.4.3-013243?logo=numpy&logoColor=white" alt="NumPy 2.4.3" />
</div>

## Project Structure

- `Simulation.py`: main training entry point.
- `Algorithms/`: RL implementations, including `AGRPO`, `CGRPO`, `PPO`, `SAC`, and `TD3`.
- `utils.py`: environment wrappers and helper utilities.
- `logger.py`: experiment logging and reproducibility support.
- `evaluate_runs.py`: metrics-to-plot analysis.
- `archive/legacy/`: archived historical variants and legacy scripts.
- `GRPO_Project.pdf`: final report.

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate GRPO_Project
```

or, if you prefer the pip path:

```bash
python -m pip install -r requirements.txt
```

### 2. Run the default training pipeline

```bash
python Simulation.py
```

This uses the default configuration in `Simulation.py`:
- environments: `dm_control/cartpole-swingup-v0`, `dm_control/acrobot-swingup-v0`, `CarRacing-v3`
- default agent: `AGRPO`
- default run mode: `full`

### 3. Customize the run

You can override the environment list and agent list with environment variables:

```bash
ENV_NAMES="dm_control/cartpole-swingup-v0,CarRacing-v3" \
AGENTS="AGRPO,PPO" \
MAX_EPISODES=200 \
MAX_STEPS=400 \
NUM_SEEDS=3 \
python Simulation.py
```

Other supported overrides from the script:

```bash
RUN_MODE="full"         # or "quick"
RECOVERY="true"         # resume from checkpoints
MULTIPROCESSING="true"  # parallel seed execution
SKIP_STEPS=4
```

## Training Pipeline

The execution flow is:

1. `Simulation.py` initializes the environment and selected agent.
2. Each seed runs one training loop with evaluation checkpoints.
3. Results are logged through `logger.py`.
4. Metrics are saved under the experiment output folders.
5. `evaluate_runs.py` can be used afterwards to produce plots from the saved results.

## Outputs

Generated artifacts are expected in:

- `checkpoints/`
- `runs/`
- `plots/`
- `backups/`
- `run_times.json`

These are experiment outputs and are intentionally kept separate from the main source code.

## Reproducibility

Each run stores metadata, checkpoints, and the source files used for that run so experiments can be traced back to the exact code version used.

## Notes

- Keep the repository focused on the final working pipeline.
- Historical or non-final variants are archived under `archive/legacy/`.
- The project is designed to be easy to run and easy to maintain for presentation and future extension.