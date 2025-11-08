# config/run.py
from dataclasses import dataclass, field
from typing import List, Optional

import torch

@dataclass(frozen=True)
class ExperimentCfg:
    name: str = "default"
    seed: int = 0
    comment: str = ""
    version_used: str = "unknown"  # git commit hash, set later

@dataclass(frozen=True)
class LandmarksCfg:
    type: str = "dots"  # options: coordinate, dots, checkerboard, gaussian, posenc_sincos_2d, posenc_fourier_features
    params: dict = field(default_factory=dict)  # parameters for the landmark function

@dataclass(frozen=True)
class ModelCfg:
    num_hidden: int = 12
    num_visible: int = 0   # derived later
    num_channels: int = 0  # derived later
    fire_rate: float = 0.5
    alive_threshold: float = 0.1
    step_size: float = 1.0
    seed_alpha: float = 1.0
    landmarks: LandmarksCfg = LandmarksCfg()  # defined later

@dataclass(frozen=True)
class TargetCfg:
    target_size: int = 32
    target_padding: int = 16
    target_image: str = ""
    target_emoji: Optional[str] = None

@dataclass(frozen=True)
class RolloutCfg:
    min: int = 64
    max: int = 96
    warmup_steps: Optional[int] = None
    starting_step_size: Optional[float] = None
    alpha_scaling: Optional[float] = None

@dataclass(frozen=True)
class SchedulerCfg:
    milestones: List[int] = field(default_factory=lambda: [2000])
    decay_factor: float = 0.1
    warmup_steps: Optional[int] = None

@dataclass(frozen=True)
class EarlyStoppingCfg:
    enabled: bool = False
    measure: str = "accuracy"  # options: loss, mse, accuracy, accuracy_snap
    warmup_steps: int = 500
    patience: int = 250
    min_delta: float = 1e-4
    window_size: int = 100

@dataclass(frozen=True)
class TrainingCfg:
    mode: str = "continuous"
    batch_size: int = 8
    pool_size: int = 1024
    pool_commit_prob: float = 1.0
    pool_commit_tolerance: float = float('inf')
    replace_with_seed: int = 1
    num_to_damage: int = 0
    learning_rate: float = 2e-3
    num_steps: int = 21
    supervise_wireworld: bool = False
    optimizer: str = "adam"
    rollout: RolloutCfg = RolloutCfg()
    epsilon: float = 1e-7
    scheduler: SchedulerCfg = SchedulerCfg()
    early_stopping: EarlyStoppingCfg = EarlyStoppingCfg()

# results, checkpoint, image and video logging

@dataclass(frozen=True)
class MetricsCfg:
    interval: int = 1
    rollout_metrics: Optional[bool] = None

@dataclass(frozen=True)
class ModelCheckpointCfg:
    interval: int = 1000

@dataclass(frozen=True)
class PoolImagesCfg:
    enabled: bool = True
    interval: int = 10
    video: bool = True
    fps: int = 10

@dataclass(frozen=True)
class BatchImagesCfg:
    enabled: bool = True
    interval: int = 10

@dataclass(frozen=True)
class CaRolloutCfg:
    enabled: bool = True
    ca_steps: int = 1200
    fps: int = 30
    use_best_checkpoint: bool = False

@dataclass(frozen=True)
class ChannelStripCfg:
    enabled: bool = False
    cmap_name: str = "viridis"
    interval: int = 1
    for_train_step: int = 1
    spacing: int = 4
    font_size: int = 8

@dataclass(frozen=True)
class ResultsCfg:
    output_dir: str = "results"
    metrics: MetricsCfg = MetricsCfg()
    model_checkpoint: ModelCheckpointCfg = ModelCheckpointCfg()
    pool_images: PoolImagesCfg = PoolImagesCfg()
    batch_images: BatchImagesCfg = BatchImagesCfg()
    ca_rollout: CaRolloutCfg = CaRolloutCfg()
    channel_strip: ChannelStripCfg = ChannelStripCfg()

# misc and compute

@dataclass(frozen=True)
class MiscCfg:
    debug: bool = False
    preview_target_image: bool = False

@dataclass(frozen=True)
class ComputeCfg:
    requested_device: str = "auto"
    device: torch.device = None
    num_threads: int = 4
    mixed_precision: bool = True

# overall configuration

@dataclass(frozen=True)
class RunCfg:
    experiment: ExperimentCfg = ExperimentCfg()
    model: ModelCfg = ModelCfg()
    target: TargetCfg = TargetCfg()
    training: TrainingCfg = TrainingCfg()
    misc: MiscCfg = MiscCfg()
    results: ResultsCfg = ResultsCfg()
    compute: ComputeCfg = ComputeCfg()
