from typing import Optional, Sequence, Union
import torch
import numpy as np
import matplotlib.cm as cm

from pathlib import Path
from PIL import Image

from src.utils.utils import ColourRGBATensor, ColourRGBTensor

def channel_strip(chw: Union[ColourRGBTensor, ColourRGBATensor], spacing:int=4, cmap_name:str='inferno', gamma:float=0.5, font_size:Optional[int]=12):
    """
    Create a horizontal strip of channels visualised using a perceptually uniform colormap.
    Args:
        chw: Tensor or ndarray of shape (C, H, W) with values in [0, 1]
        spacing: pixels between images
        cmap_name: any matplotlib colormap name
        gamma: gamma correction factor to enhance low values (0.3–0.7 is good)
    """
    if isinstance(chw, torch.Tensor):
        chw = chw.detach().cpu().numpy()

    height, width = chw.shape[1:3]
    total_width = width * chw.shape[0] + spacing * (chw.shape[0] + 1)
    strip = Image.new('RGBA', (total_width, height), (255, 255, 255, 255))

    cmap = cm.get_cmap(cmap_name)

    x_offset = spacing
    for c in range(chw.shape[0]):
        img = np.clip(chw[c], 1e-6, 1.0)
        img = img**gamma  # gamma correction

        img_rgb = (cmap(img)[..., :3] * 255).astype(np.uint8)  # drop alpha
        img_pil = Image.fromarray(img_rgb, mode='RGB').convert('RGBA')
        strip.paste(img_pil, (x_offset, 0))
        x_offset += width + spacing

    return strip

# generate a header row for the channel strip that labels each channel
# for the discrete channels, it will show their colour from the palette
# for discrete + 1 it will show A
# for the remainder it will show the channel number
def channel_headers(num_channels: int, num_visible: int, width: int, palette: Optional[Sequence[int]]=None, spacing:int=4, font_size:int=12):
    """
    Generate a header row for the channel strip.
    Args:
        num_channels: number of channels
        palette: list of RGB tuples for discrete channels
        spacing: pixels between images
        font_size: size of the font for the channel labels
    """
    from PIL import ImageDraw, ImageFont

    total_width = width * num_channels + spacing * (num_channels + 1)
    header = Image.new('RGBA', (total_width, width + 2 * spacing), (255, 255, 255, 255))
    draw = ImageDraw.Draw(header)

    font = ImageFont.load_default(size=font_size)

    x_offset = spacing
    for c in range(num_channels):
        if palette is not None:
            color = (palette[c * 3], palette[c*3+1], palette[c*3+2]) if c < num_visible else (255, 255, 255)
            draw.rectangle([x_offset, spacing, x_offset + width, width + spacing], fill=color)
            label = f"{c}"
        else:
            label = f"{c}"

        draw.text((x_offset + 2, spacing), label, fill=(0, 0, 0), font=font)
        x_offset += width + spacing

    return header

def generate_channel_grid(
    input_dir: Path = Path('images/channels'),
    output_path: Path = Path('images/channels/channel_grid.png'),
    spacing: int = 4
):
    """
    Generate a grid of channel images from the input directory.
    """

    # check how many channel images are in the directory
    channels_dir = Path(input_dir)
    channel_files = list(channels_dir.glob('*.png'))
    if not channel_files:
        return None
    channel_files.sort()  # Sort files by name
    num_channels = len(channel_files)
    print(f"Found {num_channels} channel images in {input_dir}")

    # stack the images vertically
    images = []
    for file in channel_files:
        img = Image.open(file).convert('RGBA')
        images.append(img)
    if not images:
        return None

    # join all the images vertically
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights) + (num_channels + 1) * spacing
    max_width = max(widths)
    grid_image = Image.new('RGBA', (max_width, total_height), (255, 255, 255, 255))
    y_offset = spacing
    for img in images:
        grid_image.paste(img, (0, y_offset))
        y_offset += img.height + spacing
    grid_image.save(output_path, format='PNG', compress_level=0)
    print(f"Channel grid saved to {output_path}")

    # delete the original channel images
    for file in channel_files:
        file.unlink()