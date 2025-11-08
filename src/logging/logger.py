from __future__ import annotations
from contextlib import contextmanager
import contextvars
import math
from pathlib import Path
from typing import Any, Optional
import json
import os
from typing import Any, Optional, Sequence
from PIL import Image
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from src.utils.utils import T_CA, frame_image, state_to_rgba, tile2d, to_rgba

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # guard if TB not installed

# --- global facade ------------------------------------------------------------

# thread/async-local context (no accidental bleed between tasks)
_LOGGER: Optional["Logger"] = None
_CTX = contextvars.ContextVar("log_ctx", default={})

def init_logger(output_dir: Path, **ctx) -> "Logger":
    """Create the singleton Logger and seed the global context."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger(Path(output_dir))
    if ctx:
        set_context(**ctx)
    return _LOGGER

def get_logger() -> "Logger":
    if _LOGGER is None:
        raise RuntimeError("init_logger(output_dir, ...) must be called first.")
    return _LOGGER

def set_context(**kv) -> None:
    """Replace/extend the current logging context (step, phase, run_id, etc.)."""
    cur = dict(_CTX.get())
    cur.update({k: v for k, v in kv.items() if v is not None})
    _CTX.set(cur)

def get_context() -> dict[str, Any]:
    return dict(_CTX.get())

def clear_context(*keys: str) -> None:
    """Clear specific keys, or everything if no keys are given."""
    cur = dict(_CTX.get())
    if keys:
        for k in keys: cur.pop(k, None)
    else:
        cur.clear()
    _CTX.set(cur)

@contextmanager
def log_context(**kv):
    """
    Context manager for scoped context, auto-restores on exit:
        with log_context(step=step, phase="rollout"):
            get_logger().log_metrics(loss=...)
    """
    token = _CTX.set({**_CTX.get(), **kv})
    try:
        yield get_logger()
    finally:
        _CTX.reset(token)
# ------------------------------------------------------------------------------
# helpers

def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _with_ctx(payload: dict) -> dict:
    # merge ts + global context (step, phase, run_id...) + payload
    from src.logging.logger import get_context  # safe local import to avoid cycles
    ctx = get_context()
    out = {"ts": _now_iso(), **ctx, **payload}
    return out

# --- Logger -------------------------------------------------------------------

class Logger:
    def __init__(self, output_dir: Path):
        self.root = output_dir
        self.metrics_path = Path(self.root / 'metrics.jsonl')
        self.checkpoints_dir = Path(self.root / 'checkpoints')
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = Path(self.root, 'images')
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.batch_images_dir = Path(self.images_dir, 'batch')
        self.batch_images_dir.mkdir(parents=True, exist_ok=True)
        self.pool_images_dir = Path(self.images_dir, 'pool')
        self.pool_images_dir.mkdir(parents=True, exist_ok=True)
        self.channel_images_dir = Path(self.images_dir, 'channels')
        self.channel_images_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir = Path(self.root, 'videos')
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.tb: Optional[SummaryWriter] = None # type: ignore

    # --- TensorBoard enable/close --------------------------------------------
    def enable_tensorboard(self, log_subdir: str = "tb", **writer_kwargs):
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard not available. Install via pip and re-run.")
        log_dir = writer_kwargs.pop("log_dir", self.root / log_subdir)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # common knobs: flush_secs=5, filename_suffix=""
        self.tb = SummaryWriter(log_dir=str(log_dir), **writer_kwargs)

    def close(self):
        if self.tb:
            self.tb.flush()
            self.tb.close()
            self.tb = None
    # -------------------------------------------------------------------------

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())  # durable enough for most runs

    def save_batch_image(self, x0: T_CA, x: T_CA, step_i: int, discrete: bool = False, num_visible: Optional[int] = None, palette: Optional[Sequence[int]] = None, alive_threshold: float = 0.1):
        if discrete:
            imgs0 = state_to_rgba(x0, palette, num_visible, alive_threshold).detach().cpu().numpy()
            imgs1 = state_to_rgba(x, palette, num_visible, alive_threshold).detach().cpu().numpy()
        else:
            imgs0 = to_rgba(x0).detach().cpu().numpy()
            imgs1 = to_rgba(x).detach().cpu().numpy()

        imgs0 = imgs0.transpose(0, 2, 3, 1)
        imgs1 = imgs1.transpose(0, 2, 3, 1)

        row0 = np.hstack(imgs0)  # (H, B*W, C)
        row1 = np.hstack(imgs1)
        vis = np.vstack([row0, row1])  # (2*H, B*W, C)

        # Determine file extension based on mode
        filename = self.images_dir / f"batch/batch_{step_i:04d}.png"

        img = Image.fromarray(np.clip(vis * 255, 0, 255).astype(np.uint8), mode="RGBA")

        # YYYYMMDD_HHMMSS
        now = datetime.now()
        dt_str = f"{now:%Y-%m-%d/%H%M%S}"

        nchw = tuple(x0.shape)
        img = frame_image(
            img,
            expand=4,
            text=f"Step {step_i}\nN{nchw[0]} C{nchw[1]} H{nchw[2]} W{nchw[3]}\n{dt_str}",
        )

        # make sure batch directory exists
        os.makedirs(self.images_dir / 'batch', exist_ok=True)

        img.save(filename, format="PNG", compress_level=0)
        
    def save_pool_image(self, pool, discrete=False, num_visible=Optional[int], palette=Optional[Sequence[int]], alive_threshold=0.1, step: Optional[int] = None):
        if step is None:
            step = get_context().get("step", 0)
        pool = pool.x[:49].cpu()  # take only first 49 for tiling
        
        with torch.no_grad():
            if discrete:
                rgba = state_to_rgba(pool, palette, num_visible, alive_threshold)
            else:
                rgba = to_rgba(pool)

        rgba = rgba.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)

        tiled_pool = tile2d(rgba)

        # clamp
        tiled_pool = np.clip(tiled_pool, 0, 1)

        img = Image.fromarray((tiled_pool * 255).astype(np.uint8), mode="RGBA")

        # YYYYMMDD_HHMMSS
        now = datetime.now()
        dt_str = f"{now:%Y-%m-%d/%H%M%S}"

        img = frame_image(
            img,
            expand=4,
            text=f"Step {step}\n{dt_str}",
        )

        img.save(self.images_dir / f'pool/{step:04d}_pool.png', format='PNG', compress_level=0)
        

    def save_model_checkpoint(self, model: torch.nn.Module, step: Optional[int] = None):
        if step is None:
          step = get_context().get("step", 0)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoints_dir / f"checkpoint_{int(step):04d}.pth")

    def save_image(self, img: Image.Image, path: Path):
        """Saves an RGBA image to the images directory."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path, format='PNG', compress_level=0)

    def log_metrics(self, **metrics):
        # numeric-only JSONL (your existing code)
        row = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                try: v = v.item()
                except Exception: continue
            if isinstance(v, (int, float)) and math.isfinite(v):
                row[k] = float(v)
        if not row:
            return
        self._append_jsonl(self.metrics_path, _with_ctx(row))

        # mirror to TensorBoard
        if self.tb is not None:
            step = int(get_context().get("step", 0))
            phase = get_context().get("phase")
            for k, v in row.items():
                tag = f"{phase}/{k}" if phase else k
                self.tb.add_scalar(tag, v, global_step=step)

    def find_best_checkpoint(self):
        """Find checkpoint with lowest MSE from logged metrics.
        Only considers steps where checkpoints were actually saved.
        
        Returns:
            tuple: (best_step, best_mse) or (None, None) if no valid data found
        """
        if not self.metrics_file.exists():
            return None, None
        
        # Find all checkpoint files that actually exist
        checkpoint_steps = set()
        if self.checkpoints_dir.exists():
            for checkpoint_file in self.checkpoints_dir.glob("checkpoint_*.pth"):
                # Extract step number from filename like "checkpoint_0000.pth"
                try:
                    step_str = checkpoint_file.stem.split('_')[1]
                    step = int(step_str)
                    checkpoint_steps.add(step)
                except (IndexError, ValueError):
                    continue
        
        if not checkpoint_steps:
            return None, None
            
        best_step = None
        best_mse = float('inf')
        
        # Only consider MSE values for steps where checkpoints exist
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'mse' in data and 'step' in data:
                        step = data['step']
                        mse = data['mse']
                        if step in checkpoint_steps and mse < best_mse:
                            best_mse = mse
                            best_step = step
                except (json.JSONDecodeError, KeyError):
                    continue
                    
        return best_step, best_mse if best_step is not None else (None, None)
    
    def save_best_checkpoint_reference(self, best_step):
        """Create a symlink or copy to mark the best checkpoint.
        
        Args:
            best_step: The step number of the best checkpoint
        """
        if best_step is None:
            return False
            
        checkpoint_file = self.checkpoints_dir / f"checkpoint_{best_step:04d}.pth"
        best_checkpoint_file = self.checkpoints_dir / "best_checkpoint.pth"
        
        if not checkpoint_file.exists():
            print(f"Warning: Checkpoint file {checkpoint_file} not found")
            return False
            
        # Remove existing best checkpoint if it exists
        if best_checkpoint_file.exists():
            best_checkpoint_file.unlink()
            
        try:
            # Create symlink to the best checkpoint
            best_checkpoint_file.symlink_to(checkpoint_file.name)
            print(f"Created best checkpoint reference: {best_checkpoint_file} -> {checkpoint_file.name}")
            return True
        except OSError:
            # Fallback to copying if symlink fails
            import shutil
            shutil.copy2(checkpoint_file, best_checkpoint_file)
            print(f"Copied best checkpoint: {checkpoint_file} -> {best_checkpoint_file}")
            return True
        
    # ---------- convenience TB helpers ---------------------------------------
    def tb_add_image(self, name: str, img, *, step: int | None = None):
        """
        img: torch Tensor (C,H,W) or (H,W,C), or PIL.Image, or numpy (H,W,C).
        Only 1 or 3 channels are supported by TB; if RGBA, drops alpha.
        """
        if self.tb is None:
            return
        import torch, numpy as np
        from PIL import Image

        # normalize to torch CHW float in [0,1]
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
            img = arr
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img)
        else:
            t = img

        if t.ndim == 2:                 # (H,W) -> (1,H,W)
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[0] in (1,3,4):   # CHW
                pass
            elif t.shape[-1] in (1,3,4):  # HWC -> CHW
                t = t.permute(2,0,1).contiguous()
            else:
                raise ValueError(f"Unsupported image shape {tuple(t.shape)}")
        else:
            raise ValueError(f"Unsupported image ndim {t.ndim}")

        # drop alpha if present
        if t.shape[0] == 4:
            t = t[:3]

        # to float [0,1]
        if t.dtype.is_floating_point:
            t = t.clamp(0,1)
        else:
            t = t.to(torch.float32) / 255.0

        if step is None:
            step = int(get_context().get("step", 0))
        phase = get_context().get("phase")
        tag = f"{phase}/{name}" if phase else name
        self.tb.add_image(tag, t, global_step=step)  # dataformats inferred (CHW)

    def tb_add_histogram(self, name: str, values, *, step: int | None = None, bins="auto"):
        if self.tb is None:
            return
        import torch
        v = values.detach().flatten() if hasattr(values, "detach") else values
        if step is None:
            step = int(get_context().get("step", 0))
        phase = get_context().get("phase")
        tag = f"{phase}/{name}" if phase else name
        self.tb.add_histogram(tag, v, global_step=step, bins=bins)

    def tb_add_hparams(self, hparams: dict, final_metrics: dict[str, float]):
        """
        Logs HParams once; TB will create a child run with an hparams summary.
        """
        if self.tb is None:
            return
        # TensorBoard expects plain scalars/strings
        clean_h = {k: (str(v) if not isinstance(v, (int,float,bool)) else v) for k,v in hparams.items()}
        clean_m = {k: float(v) for k, v in final_metrics.items()}
        self.tb.add_hparams(clean_h, clean_m)
    # -------------------------------------------------------------------------