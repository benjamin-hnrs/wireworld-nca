import random
import numpy as np
import torch
from src.config.finalise import finalise
from src.config.run import RunCfg
from src.logging.logger import set_context
from src.nca.nca import NCA
from src.training.base_trainer import NCATrainer
from src.training.callbacks import CheckpointCallback, LoggingCallback, VisualisationCallback
from src.training.discrete_trainer import DiscreteTrainer
from src.training.continuous_trainer import ContinuousTrainer
from src.config.io import load_cfg
from src.utils.utils import save_config

def setup_environment(cfg):
    torch.set_num_threads(cfg.compute.num_threads)
    seed = int(cfg.experiment.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 

def setup_trainer(config: RunCfg, model: torch.nn.Module) -> NCATrainer:
    if config.training.mode == "continuous":
        trainer = ContinuousTrainer(model=model, config=config)
    else:
        trainer = DiscreteTrainer(model=model, config=config)

    trainer.add_callback(LoggingCallback()) # vis depends on logger being created
    trainer.add_callback(VisualisationCallback())
    trainer.add_callback(CheckpointCallback())

    return trainer

def setup_model(config: RunCfg) -> torch.nn.Module:
    return NCA(
        config=config,
        num_visible=config.model.num_visible,
        num_hidden=config.model.num_hidden,
        device=config.compute.device,
        fire_rate=config.model.fire_rate,
        alive_threshold=config.model.alive_threshold,
        default_step_size=config.model.step_size,
    ).to(config.compute.device)


def main():
    cfg = finalise(load_cfg("configs/base_config.yaml"))
    setup_environment(cfg)
    model = setup_model(cfg)
    trainer = setup_trainer(cfg, model)

    print(f'{"Running experiment:":<25} {cfg.experiment.name}')
    print(f'{"Device:":<25} {cfg.compute.device}')

    save_config(cfg, cfg.results.output_dir)

    trainer.train()

if __name__ == "__main__":
    main()
