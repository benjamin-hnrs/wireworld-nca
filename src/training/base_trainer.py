from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import tqdm
from abc import abstractmethod
from src.config.run import RunCfg
from src.training.sample_pool import SamplePool
from src.utils import landmarks
import bisect

from src.utils.utils import T_CA, load_palette_from_image, state_to_rgba, to_rgba
from src.nca.nca import NCA
from src.visualisation.channel_grid import channel_headers, channel_strip
from src.logging.logger import set_context, get_logger

class NCATrainer:
    def __init__(self, model: NCA, config: RunCfg):
        self.config = config
        self.device = torch.device(self.config.compute.device)
        self.mode = self.config.training.mode

        self.model = (
            model.to(self.device)
            if self.device.type == "mps"
            else torch.compile(model.to(self.device))
        )

        self.lr = float(self.config.training.learning_rate)
        self.batch_size = self.config.training.batch_size
        self.pool_size = self.config.training.pool_size
        self.steps = self.config.training.num_steps
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr, eps=float(self.config.training.epsilon))

        self.scheduler = self._init_scheduler()
        self.callbacks = []
        # if the target is a discrete image, we'll get the palette, even for a continuous model
        # because we can use it to compare accuracy with the discrete model by snapping colours
        # to the palette
        # if target_image is not specified, we assume an emoji target is used
        if self.config.target.target_image:
            self.palette = load_palette_from_image(Path(self.config.target.target_image))
        else:
            # dummy palette
            self.palette = torch.tensor([[0,0,0,0],[1,1,1,1]], dtype=torch.float32)

        self.early_stopping = self.config.training.early_stopping.enabled
        if self.early_stopping:
            self.early_stopping_warmup = self.config.training.early_stopping.warmup_steps
            self.patience = self.config.training.early_stopping.patience
            self.min_delta = float(self.config.training.early_stopping.min_delta)
            self.window_size = self.config.training.early_stopping.window_size

        self.setup()

    def _init_scheduler(self):
        milestones = self.config.training.scheduler.milestones
        gamma = self.config.training.scheduler.decay_factor

        def lr_lambda(step: int):
            if step < milestones[0]:
                return 1.0
            k = bisect.bisect_right(milestones, step)
            return gamma ** k

        return optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda)

    @abstractmethod
    def setup(self):
        pass

    def add_callback(self, callback):
        self.callbacks.append(callback)

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass

    def _seed(self, discrete=False, seed_value=1.0, seed_pos="centre"):
        C = self.model.num_channels
        _, _, H, W = self.target.shape
        a = self.model.num_visible          # alpha starts after visible channels

        if seed_pos == "centre":
            px, py = W // 2, H // 2
            print(f"Seeding at centre: {(px,py)}")
        elif seed_pos == "random":
            px = np.random.randint(W // 4, 3 * W // 4)
            py = np.random.randint(H // 4, 3 * H // 4)
            print(f"Seeding at random: {(px,py)}")
        else:
            px, py = seed_pos
            print(f"Seeding at specified position: {(px,py)}")

        seed = torch.zeros((C, H, W),
                        dtype=torch.float32,
                        device=self.device)
        
        if discrete:
            seed[a - 1, py, px] = 1.0       # set last colour index so seed is visible
        seed[a:, py, px] = seed_value       # set alpha + hidden channel values

        return seed

    def _damage_batch(self, x0: torch.Tensor, num_to_damage: int) -> torch.Tensor:
        if num_to_damage > 0:
            damage = 1.0 - self._make_circle_masks(num_to_damage, x0.shape[2], x0.shape[3]).to(x0.device)
            damage = damage[:, None, :, :]  # add channel dimension
            x0[-num_to_damage:] *= damage   # apply to the last n in batch
        return x0
    
    def _make_circle_masks(self, num_masks: int, height: int, width: int, radius: float = 0.4) -> torch.Tensor:
        x = torch.linspace(-1.0, 1.0, width)[None, None, :]             # normalised grid
        y = torch.linspace(-1.0, 1.0, height)[None, :, None]
        centre = torch.empty([2, num_masks, 1, 1]).uniform_(-0.5, 0.5)  # random centres for each mask
        r = torch.empty(num_masks, 1, 1).uniform_(0.1, radius)          # random radii
        x, y = (x - centre[0])/r, (y - centre[1])/r                     # offset and scale by radius
        mask = (x*x + y*y < 1.0).float()                                # circle test
        return mask                                                     # (n, h, w)

# ==============================================================================
# helpers
# ==============================================================================

    @torch.no_grad()
    def _init_pool(self):
        x0 = (
            self.seed.unsqueeze(0)
            .cpu()
            .numpy()
            .repeat(self.config.training.pool_size, axis=0)
        )
        return SamplePool(x=x0, device=self.device)


    def _random_rollout_iter(self) -> int:
        min = self.config.training.rollout.min
        max = self.config.training.rollout.max
        iter_n = torch.randint(min, max, (1,)).item()
        return int(iter_n)
    
    def _get_batch(self, pool):
        if self.config.training.pool_size > 0:
            batch = pool.sample(self.config.training.batch_size)
            x0 = batch.x

            with torch.no_grad():
                losses = self.loss(x0)
            loss_rank = torch.argsort(losses, descending=True)

            # replace the worst n patterns with the seed
            n = self.config.training.replace_with_seed
            x0 = x0[loss_rank]
            x0[:n] = self.seed.unsqueeze(0).repeat(n, 1, 1, 1)
        else:
            batch = None
            x0 = self.seed_batch
        return batch,x0
    

# ==============================================================================
# training
# ==============================================================================
    
    def rollout(self, x, t):
        # if isinstance(x, np.ndarray):
        #     print("Converting numpy array to tensor in train_step!")
        #     x = torch.tensor(x, dtype=torch.float32, device=self.device)

        for cb in self.callbacks:
            cb.on_rollout_start(self, t)

        iter_n = self._random_rollout_iter()
        x = x.contiguous() # TODO do i need to assign this back to x?
        x_seq = [x]        # TODO this is used to keep track of the steps for WW sim loss

        
        for i in range(iter_n):
            for cb in self.callbacks:
                cb.on_rollout_step_start(self, t, i, x)
            # handle ca step size warmup
            sss = self.config.training.rollout.starting_step_size if self.config.training.rollout.starting_step_size else 1.0
            if self.config.training.rollout.warmup_steps:
                if self.config.training.rollout.warmup_steps > 0:
                    step_size = min(1.0, sss + (1.0-sss) * i / self.config.training.rollout.warmup_steps)
                else:
                    step_size = 1.0

            x = self.model(x)

            if self.config.training.supervise_wireworld:
                x_seq.append(x)
                # x_seq.append(x.clone()) # keep grad

            for cb in self.callbacks:
                cb.on_rollout_step_end(self, t, i, x)


        # which loss do we use?
        if self.config.training.supervise_wireworld:
            loss = self._compute_wireworld_loss(x_seq)
        else:
            loss = self.loss(x).mean()

        self.optimiser.zero_grad()
        loss.backward()
        self._normalise_gradients()
        self.optimiser.step()
        if self.scheduler:
            self.scheduler.step()

        for cb in self.callbacks:
            cb.on_rollout_end(self, t, i, x)

        return x.detach(), loss.item()

    def _normalise_gradients(self):
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    # denom = p.grad.norm().clamp_min(1e-8)
                    denom = p.grad.norm() + 1e-8
                    p.grad.div_(denom)
    
    
    def train(self):
        for cb in self.callbacks:
            cb.on_train_start(self)

        self.model.train()
        pool = self._init_pool()
        loss_log = []

        # tracking for early stopping
        if self.early_stopping:
            self.recent_es_measures = []
            best_avg = float("inf")
            steps_since_improvement = 0
        
        for i in tqdm.trange(self.steps, desc="Training"):
            for cb in self.callbacks:
                cb.on_train_step_start(self, i)

            batch, x0 = self._get_batch(pool)

            x0 = self._damage_batch(x0, self.config.training.num_to_damage)

            step_i = len(loss_log)

            x, loss = self.rollout(x0, step_i)

            # return the updated batch to the pool
            if self.config.training.pool_size > 0:
                # mask if all cells are dead (alpha channel is below threshold everywhere)
                mask = (x[:, self.model.num_visible:self.model.num_visible+1] > self.config.model.alive_threshold).any(dim=(1,2,3))

                if mask.any():
                    batch.x[mask] = x[mask]
                batch.commit(pool, batch._parent_idx)

            loss_log.append(loss)


            if self.early_stopping:
                self.recent_es_measures.append(loss)

                if len(self.recent_es_measures) > self.window_size:
                    self.recent_es_measures.pop(0)

                if i >= self.early_stopping_warmup and len(self.recent_es_measures) == self.window_size:
                    avg = np.mean(self.recent_es_measures)
                    if avg < best_avg - self.min_delta:
                        best_avg = avg
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1

                    if steps_since_improvement >= self.patience:
                        print(f"Early stopping at step {i} (no improvement in moving average {loss} ({avg}) for {self.patience} steps)")
                        break

            for cb in self.callbacks:
                cb.on_train_step_end(self, i, pool, x0, x, loss)

        for cb in self.callbacks:
            cb.on_train_end(self, i)
        return loss_log