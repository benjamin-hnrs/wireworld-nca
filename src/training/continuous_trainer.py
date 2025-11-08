import torch
import torch.nn.functional as F
from pathlib import Path
from src.config.run import RunCfg
from src.nca.nca import NCA
from src.training.base_trainer import NCATrainer
from src.utils.utils import (
    load_resize_and_pad_image,
    load_resize_and_pad_emoji,
    to_rgba,
)


class ContinuousTrainer(NCATrainer):
    def __init__(self, model: NCA, config: RunCfg):
        super().__init__(model, config)

    def setup(self):
        print("Setting up ContinuousTrainer...")
        # if the config has the emoji attribute
        if self.config.target.target_emoji:
            self.target = (
                load_resize_and_pad_emoji(
                    self.config.target.target_emoji,
                    target_size=self.config.target.target_size,
                    padding=self.config.target.target_padding,
                )
                .to(self.device)
                .float()
            )
        else:
            self.target = (
                load_resize_and_pad_image(
                    Path(self.config.target.target_image),
                    target_size=self.config.target.target_size,
                    padding=self.config.target.target_padding,
                )
                .to(self.device)
                .float()
            )

        self.target = self.target.unsqueeze(0)

        self.seed = self._seed(seed_value=self.config.model.seed_alpha)
        self.seed_batch = self.seed.unsqueeze(0).repeat(self.config.training.batch_size, 1, 1, 1)

    def loss(self, x):
        x_rgba = to_rgba(x)                         # keep only the RGBA channels (N, 4, H, W)
        diff = x_rgba - self.target

        # original gnca used
        # return tf.reduce_mean(tf.square(to_rgba(x)-pad_target), [-2, -3, -1])

        return torch.mean(diff**2, dim=[1, 2, 3])   # mse for each batch item (N items)