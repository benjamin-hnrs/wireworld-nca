# Growing Wireworld circuits with Neural Cellular Automata

![Neural cellular automata growing a discrete Wireworld pattern that simulates a single pixel of Langton's Ant](nca.gif)

This repository contains an independent PyTorch implementation of the NCA framework from [Growing Neural Cellular Automata]([Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)) by Mordvintsev et al. (2020)

It has been extended with a training method for learning discrete images, as well as some positional scaffolding mechanisms.

There is a small dataset of [Wireworld](https://en.wikipedia.org/wiki/Wireworld) images included.

## Quick start

1. Install dependencies
   - Create a virtual environment and install:
     pip install -r requirements.txt

2. Configure training run
   - The training entrypoint loads a config via `load_cfg("configs/base_config.yaml")`.
   - Create or edit `configs/base_config.yaml` (or place a copy of any reproduction config here) to set:
     - target image
     - training parameters (mode: `discrete` or `continuous`, batch_size, pool_size, learning_rate, rollout settings, early stopping, etc.)
     - compute.device settings (e.g. `auto`, `cpu`, `cuda`, `mps`)
     - results.output_dir where outputs, logs, checkpoints and images will be written

3. Run training
   - Run the main training script (`run.py`):
     `python -m run`

4. Where outputs go
   - The run output directory is set via the config (`results.output_dir`). Within it you will find:
     - model checkpoints
     - batch and sample pool images
     - video
     - config.yaml (saved copy of finalised config)
     - metrics.jsonl, events and other logs

## Reproducing experiments

- Reproduction configs are available in the `reproduction/` dir.
  - Copy the desired run config `reproduction/<run>/config.yaml` into `configs/base_config.yaml`, or use it as the config file path when loading (the code uses `configs/base_config.yaml` by default).
  - Example: copy `reproduction/some_run/config.yaml` -> `configs/base_config.yaml`, then run `python scripts/main.py`.
  - Note that the `inference-gnca` folder contains output from the [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (used when validating implementation), not this PyTorch implementation.

- Evaluation of trained checkpoints:
  - Use [analysis/evaluate_checkpoints.py](analysis/evaluate_checkpoints.py) to evaluate checkpoints in a results subdirectory. That script will load `config.yaml` from each run subdir and run inference/evaluation metrics.

## Tips
- Using the discrete training mode:
  - The target image must be an palette indexed  image. The first colour will be interpreted as dead cells, regardless of its transparency. Transparency in other palette entries is ignored.
  - `num_visible` is automatically set for discrete mode.

## Project layout (directory contents)

- src/
  - Primary Python package containing model, training loop, config handling and utilities. Importable as the repository package root used by most scripts.
- configs/
  - Default and example YAML configuration files. Copy or edit `configs/base_config.yaml` for runs.
- reproduction/
  - Saved reproduction experiments. Each subdirectory contains a finalised `config.yaml`. Use these configs to reproduce published runs.
- results/
  - Default output location for training runs (checkpoints, images, videos, metrics). Created at runtime per experiment.
- analysis/
  - Scripts and notebooks for evaluating checkpoints, plotting results and running post-hoc analyses (very messy code here)
- requirements.txt
  - Python dependencies used to run and reproduce experiments.
- setup.py
  - Installable package metadata / entry points (if you prefer pip install -e .).

## References
### Growing Neural Cellular Automata
This implementation is based on the original TensorFlow by Mordvintsev et al. available here: [Growing Neural Cellular Automata]([Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)).

The video output code is taken directly from their implementation.

### Wireworld target images
Most of the Wireworld images were taken from [Golly](https://golly.sourceforge.io). The [Wireworld computer](https://www.quinapalus.com/wi-index.html) (also available in Golly) is by David Moore, Mark Owen and others  **golly-nick-gardner-256.png** is a multiplier by Nick Gardner, also found here [Math Games: WireWorld Multiplication](https://www.mathpuzzle.com/MAA/20-WireWorld/mathgames_05_24_04.html).