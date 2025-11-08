import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.run import RunCfg
from src.utils.landmarks import make_landmark_fn

from src.logging.logger import get_logger

class NCA(nn.Module):
    def __init__(self,
                 config: RunCfg,
                 num_visible: int,
                 num_hidden: int,
                 device: torch.device,
                 fire_rate: float = 0.5,
                 alive_threshold: float = 0.1,
                 default_step_size: float = 1.0):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.num_channels = self.num_visible + self.num_hidden + 1  # + 1 for alpha

        self.device = device

        self.num_kernels = self.initialise_kernels()
        self.kernel_size = 3
        self.kernel_padding = 1
        self.stride = 1
        
        self.fire_rate = float(fire_rate)
        self.alive_threshold = float(alive_threshold)
        self.default_step_size = float(default_step_size)

        self.landmarks = make_landmark_fn(config) # TODO can we just pass the model config?
        landmark_channels = self.landmarks.shape[0] if self.landmarks is not None else 0

        # CA update rule
        self.dmodel = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels * self.num_kernels + landmark_channels,
                      out_channels=128, kernel_size=1),
            nn.ReLU(),

            # TODO: REMOVE ME!
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            # nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=self.num_channels,
                      kernel_size=1, bias=False)
        )
        
        # initialise network
        with torch.no_grad():
            nn.init.xavier_uniform_(self.dmodel[0].weight)
            nn.init.zeros_(self.dmodel[0].bias)

            # TODO: REMOVE ME!
            # nn.init.xavier_uniform_(self.dmodel[2].weight)
            # nn.init.zeros_(self.dmodel[2].bias)

            nn.init.zeros_(self.dmodel[-1].weight)

    
    def initialise_kernels(self):
        identity = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=torch.float32)

        dx = torch.tensor([
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1]
        ], dtype=torch.float32) / 8.0

        dy = dx.T

        avg = torch.ones((3, 3), dtype=torch.float32) / 9.0

        lap_5 = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32)

        lap_9 = torch.tensor([
            [1,   2, 1],
            [2, -12, 2],
            [1,   2, 1]
        ], dtype=torch.float32)

        kernels_list = [identity, dx, dy]
        kernel_names = ["identity", "dx", "dy"]

        # stack all kernels into (num_kernels, 3, 3)
        kernels = torch.stack(kernels_list, dim=0)

        # expand for depthwise conv2d: (num_kernels * num_channels, 1, 3, 3)
        kernels = kernels.unsqueeze(1)                       # (nk, 1, 3, 3)
        kernels = kernels.repeat(self.num_channels, 1, 1, 1) # (nk * C, 1, 3, 3)
        kernels = kernels.to(self.device)

        self.register_buffer("perception_kernels", kernels)

        return len(kernels_list)
    

    def get_living_mask(self, x):
        """
        Returns a mask indicating which cells might need to be updated
        (i.e. those with living neighbours). A cell is considered alive if
        it has any neighbours (kernel_size) with an alpha value greater than
        the alive_threshold.
        """
        a = self.num_visible   # alpha channel follows the rgb/colour index channels
        alpha = x[:, a:a+1, :, :]
        alive = F.max_pool2d(
            alpha,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_padding
        )
        return alive > self.alive_threshold


    def perceive(self, x):
        assert x.dim() == 4, "Input tensor must be 4D (N, C, H, W)"
        assert x.size(1) == self.num_channels, f"Input tensor must have {self.num_channels} channels, got {x.size(1)}"

        return F.conv2d(x, self.perception_kernels, padding=1, groups=self.num_channels)


    def forward(self, x, fire_rate=None, step_size=1.0):
        pre_mask = self.get_living_mask(x)                 # find cells that are alive (or neighbouring alive cells)
        y = self.perceive(x)                               # get perception for all cells        
        if self.landmarks is not None:
            # TODO register this so it's only created once
            y = torch.cat([y, self.landmarks.expand(x.size(0), -1, -1, -1)], dim=1)  
        dx = self.dmodel(y) * step_size                    # predict update/delta for all cells (scaled by step_size)  # TODO: import step size from config
        
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        update_mask = (torch.rand_like(x[:, :1, :, :]) <= fire_rate).float()  # random mask for updates

        x = x + dx * update_mask                           # apply delta to randomly selected cells

        post_mask = self.get_living_mask(x)                # find cells that are alive after the update
        life_mask = (pre_mask & post_mask).float()

        return x * life_mask                               # filter out dead cells
