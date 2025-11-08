from typing import Union
import torch
import numpy as np
from pathlib import Path
from src.config.run import RunCfg
from src.logging.logger import get_logger
from src.nca.nca import NCA
from src.training.base_trainer import NCATrainer
from src.utils.utils import CAStateOneHotTensor, load_indexed_target_image, split_classes_alpha

import torch.nn.functional as F

def top1_logistic_loss(logits, y_idx, alive_mask=None, ignore_index=255, temperature=1.0):
    """
    Smooth version: loss = softplus((max_neg - t)/T) * T
    """
    N, K, H, W = logits.shape
    valid = (y_idx != ignore_index)
    if alive_mask is not None:
        valid = valid & (alive_mask.squeeze(1) > 0)

    t = logits.gather(1, y_idx.clamp_min(0).unsqueeze(1)).squeeze(1)  # (N,H,W)
    neg = logits.clone()
    neg.scatter_(1, y_idx.clamp_min(0).unsqueeze(1), float('-inf'))   # replace the true class logit with -inf
    max_neg = neg.amax(dim=1)

    d = (max_neg - t) / temperature
    loss_map = F.softplus(d) * temperature  # ~ log(1+exp(max_neg - t))

    loss = (loss_map * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    return loss


# def top1_hinge_loss(logits, y_idx, alive_mask=None, ignore_index=255, margin=0.0):
#         """
#         logits: (N, K, H, W) raw scores
#         y_idx:  (N, H, W) int in [0..K-1] or ignore_index
#         alive_mask: (N,1,H,W) 1 where supervised, else 0  (optional)
#         margin: >=0; 0 means “just be the top”, >0 enforces a small gap
#         """
#         N, K, H, W = logits.shape
#         # valid supervision
#         valid = (y_idx != ignore_index)
#         if alive_mask is not None:
#             valid = valid & (alive_mask.squeeze(1) > 0)

#         # true-class logit
#         t = logits.gather(1, y_idx.clamp_min(0).unsqueeze(1)).squeeze(1)  # (N,H,W)

#         # max negative logit (exclude the true class)
#         neg = logits.clone()
#         neg.scatter_(1, y_idx.clamp_min(0).unsqueeze(1), float('-inf'))
#         max_neg = neg.amax(dim=1)  # (N,H,W)

#         # hinge: relu(margin + max_neg - t)
#         loss_map = F.relu(margin + max_neg - t)

#         # average over valid pixels
#         loss = (loss_map * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
#         return loss

class DiscreteTrainer(NCATrainer):

    def __init__(self, model: NCA, config: RunCfg):
        super().__init__(model, config)

    def setup(self): 
        print("Setting up DiscreteTrainer...")

        # TODO we're getting the palette twice (once in base_trainer, once here)
        self.target, self.palette = load_indexed_target_image(
            Path(self.config.target.target_image),
            padding=self.config.target.target_padding,
            num_visible=self.model.num_visible,
            alive_threshold=self.model.alive_threshold,
            device=self.device
        ) # 1, C, H, W

        # should already be on the correct device
        self.target = self.target.to(self.device).float()

        self.seed = self._seed(discrete=True, seed_value=self.config.model.seed_alpha)
        self.seed_batch = self.seed.unsqueeze(0).repeat(self.config.training.batch_size, 1, 1, 1)

    def loss(
        self,
        x,                      # (N, K+1, H, W): K class logits + 1 alpha logit
        alpha_is_logit: bool = False,
        alpha_weight: float = 0.25,     # TODO: what should this be set to?
        alpha_pos_weight: Union[float,None] = None,   # e.g. >1.0 to upweight opaque pixels
        alive_from_classes: bool = False,         # derive alive mask from classes
        alpha_thr_prob: float = 0.1              # only used if alive_from_classes=False
    ):
        
        K = self.model.num_visible
        N, C, H, W = x.shape
        target = self.target.expand(N, -1, -1, -1)
        assert C >= K+1
        x_classes, x_alpha = x[:, :K], x[:, K:K+1]
        y_classes, y_alpha = target[:, :K], target[:, K:K+1]

        # --- build alive mask ---
        if alive_from_classes:
            # robust: alive where any class channel is 1
            alive = (y_classes.sum(dim=1, keepdim=True) > 0).float()  # (1,1,H,W)
        else:
            if alpha_is_logit:
                alive = (torch.sigmoid(y_alpha) > 0.5).float()
            else:
                alive = (y_alpha > alpha_thr_prob).float()            # (1,1,H,W)

        # broadcast to batch
        alive_b = alive.expand(N, 1, H, W)

        # --- CE over classes on alive pixels only ---
        # targets in [0..K-1]
        y_idx = y_classes.argmax(dim=1)        # (1,H,W)
        y_idx_b = y_idx.expand(N, H, W)        # (N,H,W)

        ce_like = top1_logistic_loss(x_classes, y_idx, alive_b, ignore_index=255, temperature=1.0)

        # --- alpha supervision everywhere, with optional pos weighting ---
        if alpha_pos_weight is not None:
            # BCEWithLogits supports pos_weight > 1 to upweight y=1
            bce = F.binary_cross_entropy(x_alpha.clamp(0,1), y_alpha, reduction="none", pos_weight=torch.tensor([alpha_pos_weight], device=x.device))
        else:
            bce = F.binary_cross_entropy(x_alpha.clamp(0,1), y_alpha, reduction="none")
        alpha_loss = bce.mean(dim=(1,2,3))  # (N,)

        # mse for alpha
        # alpha_mse = F.mse_loss(x_alpha, y_alpha, reduction="none")
        # alpha_loss = alpha_mse.mean(dim=(1, 2, 3))  # (N,)

        total = (1.0 - alpha_weight) * ce_like + alpha_weight * alpha_loss
        return total#, {"ce": ce_loss.mean().item(), "alpha": alpha_loss.mean().item()}