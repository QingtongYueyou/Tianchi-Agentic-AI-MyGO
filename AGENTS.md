# Project Memory

- Use the `minimind` conda environment for project commands, tests, training, and simulations unless the user says otherwise.
- Run tests with `conda run -n minimind python -m pytest`.
- On Windows, avoid running multiple `conda run -n minimind ...` commands in parallel because conda can collide on temp activation files.
- RL training from the project root: `conda run -n minimind python demo/agent/train.py --phase 2`.
- RL training from the `demo/` directory: `conda run -n minimind python agent/train.py --phase 2`.
- Short RL smoke training: `conda run -n minimind python demo/agent/train.py --phase 2 --episodes 1`.
- Per-driver RL fine-tuning from the project root: `conda run -n minimind python demo/agent/train.py --phase 3`.
- RL training reads `demo/agent/configs/rl_config.yaml` by default and writes policy weights to `demo/agent/models/`.
