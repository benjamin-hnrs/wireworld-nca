from pathlib import Path
from typing import Optional, Sequence, Union, cast
from omegaconf import DictConfig, ListConfig
import imageio
import torch

from src.utils.utils import CAStateOneHotTensor, CAStateRGBTensor, ColourRGBTensor, state_to_rgba, to_rgba
from src.nca.nca import NCA
from src.utils import landmarks
import numpy as np

def generate_nca_video(
    config: Union[ListConfig, DictConfig],
    checkpoint_path: Path,
    output_path: Path = Path("nca_growth.mp4"),
    grid_size: int = 256,
    num_channels: int = 16,
    n_steps: int = 200,
    fps: int = 30,
    scale: int = 2,
    seed_position: str = "centre",
    discrete: bool = False,
    palette: Optional[Sequence[int]] = None,
):
    """
    Generate an animated video of NCA growth from a checkpoint file.
    """

    from src.utils.utils import nchw_zoom
    from pathlib import Path
    import torch
    import numpy as np

    # TODO make this use mps if possible
    device = config.compute.device
    
    model = NCA(
        config=config,
        num_visible=config.model.num_visible,
        num_hidden=config.model.num_hidden,
        device=device,
        fire_rate=config.model.fire_rate,
        alive_threshold=config.model.alive_threshold,
        default_step_size=config.model.step_size
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device) # type: ignore
    model.load_state_dict(checkpoint)
    model.eval()

    seed = _seed(chw=(num_channels, grid_size, grid_size), alpha_index=config.model.num_visible, device=config.compute.device, seed_value=config.model.seed_alpha, seed_pos=seed_position, discrete=discrete)
    x = seed.unsqueeze(0)

    writer = imageio.get_writer(output_path, fps=fps) # type: ignore

    print(f"Generating {n_steps} frames...")
    with torch.no_grad():
        for step in range(n_steps):
            if discrete:
                assert palette is not None, "Palette must be provided for discrete models"
                x = cast(CAStateOneHotTensor, x)
                # Convert logits to RGB using the palette
                rgba = state_to_rgba(x, palette, config.model.num_visible, config.model.alive_threshold)
            else:
                pass
                # # Get RGB from NCHW tensor
                # x_sample = cast(CAStateRGBTensor, x[0:1])  # (1, 3, H, W) or (B, 3, H, W)?
                # rgba = to_rgb(x_sample)
                rgba = to_rgba(x)

            rgba: ColourRGBTensor = cast(ColourRGBTensor, rgba[0])  # (3, H, W)

            if scale > 1:
                rgba = nchw_zoom(rgba, scale)  # Still CHW

            # to np
            rgb_arr = rgba.cpu().numpy() # type: ignore

            frame = rgb_arr.transpose(1, 2, 0)  # CHW → HWC # type: ignore
            frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
            writer.append_data(frame_uint8)

            # Update NCA
            x = model(x)

            if (step + 1) % 20 == 0:
                print(f"  Frame {step + 1}/{n_steps}")

    writer.close()
    print(f"Video saved to: {output_path}")
    return Path(output_path)


def _seed(chw, alpha_index, device, discrete=False, seed_value=1.0, seed_pos="centre"):
    C, H, W = chw
    a = alpha_index          # alpha starts after visible channels

    if seed_pos == "centre":
        px, py = W // 2, H // 2
    elif seed_pos == "random":
        px = np.random.randint(W // 4, 3 * W // 4)
        py = np.random.randint(H // 4, 3 * H // 4)
    else:
        px, py = seed_pos

    seed = torch.zeros((C, H, W),
                    dtype=torch.float32,
                    device=device)
    
    if discrete:
        seed[a - 1, py, px] = 1.0       # set last colour index so seed is visible
    seed[a:, py, px] = seed_value       # set alpha + hidden channel values

    return seed