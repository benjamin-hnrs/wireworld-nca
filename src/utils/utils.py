import io
import os
from pathlib import Path
from urllib.parse import urlparse
import requests
import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import Any, NewType, Optional, TypeVar, Union, cast


# ------------------------------------------------------------------------------
# Load emoji/resize
# ------------------------------------------------------------------------------

def get_palette_size(img_path: Path) -> int:
    """Get the number of discrete channels based on the palette size."""
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = Image.open(img_path)

    if img.mode != "P":
        raise ValueError(f"Image {img_path} is not in palette mode (mode={img.mode})")

    palette = img.getpalette()

    if not palette:
        raise ValueError(f"No palette found in image: {img_path}")
    
    palette_size = len(palette) // 3
    
    print(f'{"Target palette size:":<25} {palette_size}')
    return palette_size

def load_image(source, max_size, device=None):
    """
    Loads an image from a file path or URL, resizes it to fit within max_size,
    premultiplies alpha and returns a tensor in HWC format with values in [0, 1].
    """
    if isinstance(source, Path):
        source = str(source)
    # check if source is a URL or file path
    parsed = urlparse(source)
    
    if parsed.scheme in ('http', 'https'):
        r = requests.get(source)
        img = PIL.Image.open(io.BytesIO(r.content))
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        img = PIL.Image.open(source)
    
    # ensure image has alpha channel
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    
    # resize
    img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255.0
    img[..., :3] *= img[..., 3:]  # premultiply rgb by alpha
    img_tensor = torch.tensor(img, dtype=torch.float32)
    if device:
        img_tensor = img_tensor.to(device)
    return img_tensor

def load_emoji(emoji, max_size=48):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url, max_size)

def _load_resize_and_pad_emoji(img, target_size: tuple, padding: int):
    img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    rgba = to_rgba(img) # (4, H, W)

    # pad starts from last dimension, left then right
    return F.pad(rgba, (padding, padding, padding, padding)) # (4, H + 2p, W + 2p)
    # TODO need to move image to correct device


def load_resize_and_pad_image(path: Path, target_size: tuple, padding: int):
    # load target image and add alpha
    img = load_image(path, max_size=target_size)
    return _load_resize_and_pad_emoji(img, target_size, padding)


def load_resize_and_pad_emoji(emoji: str, target_size: tuple, padding: int):
    img = load_emoji(emoji, max_size=target_size)
    return _load_resize_and_pad_emoji(img, target_size, padding)

def to_rgba(x):  
    if x.ndim == 3:
        if x.shape[0] < 4:
            raise ValueError(f"Input x has too few channels ({x.shape[0]}), expected at least 4.")
        return x[:4, :, :]  # (4, H, W)
    elif x.ndim == 4:
        if x.shape[1] < 4:
            raise ValueError(f"Input x has too few channels ({x.shape[1]}), expected at least 4.")
        return x[:, :4, :, :]  # (B, 4, H, W)
    else:
        raise ValueError(f"Expected 3D (C,H,W) or 4D (N,C,H,W) NCHW tensor, got shape {x.shape}")



# ------------------------------------------------------------------------------
# Index/state/RGBA conversions
# ------------------------------------------------------------------------------

# {indexed, one-hot, rgb} + alpha OR rgb; no hidden channels
ColourIdxTensor = NewType("ColourIdxTensor", torch.Tensor)  # N, H, W
ColourOneHotAlphaFirstTensor = NewType("ColourOneHotAlphaFirstTensor", torch.Tensor)  # N, C, H, W
ColourOneHotAlphaLastTensor = NewType("ColourOneHotAlphaLastTensor", torch.Tensor)  # N, C, H, W
ColourRGBTensor = NewType("ColourRGBTensor", torch.Tensor)  # N, 3, H, W
ColourRGBATensor = NewType("ColourRGBATensor", torch.Tensor)  # N, 4, H, W

# {indexed, rgb colour} + alpha + hidden channels
CAStateOneHotTensor = NewType("CAStateOneHotTensor", torch.Tensor)  # N, P+A+H, H, W
CAStateRGBTensor = NewType("CAStateRGBTensor", torch.Tensor)  # N, 3+A+H, H, W

# images
IndexImage = NewType("IndexImage", PIL.Image.Image)  # H, W, P
RGBImage = NewType("RGBImage", PIL.Image.Image)  # H, W, 3
RGBAImage = NewType("RGBAImage", PIL.Image.Image)  # H, W, 4

T_RGB_A = TypeVar("T_RGB_A", ColourRGBTensor, ColourRGBATensor)
T_ANYRGB_A = TypeVar("T_ANYRGB_A", ColourRGBTensor, ColourRGBATensor, CAStateRGBTensor)
T_CA = TypeVar("T_CA", CAStateOneHotTensor, CAStateRGBTensor)

# Channel layout: [ K class logits, 1 alpha, (hidden...) ]
# where K = num_visible (visible classes only; transparent is NOT one of them)

def split_classes_alpha(x, num_visible: int):
    """Splits (N, C, H, W) into class and alpha components."""
    classes = x[:, :num_visible]                 # (N, K, H, W)
    alpha   = x[:, num_visible:num_visible+1]   # (N, 1, H, W)
    return classes, alpha

@torch.no_grad()
def state_to_indices(x, num_visible: int, alive_threshold: float):
    """
    Convert model state -> palette indices where:
      - alpha < threshold => 0 (transparent)
      - else => argmax over K classes, shifted +1 to avoid index 0
    """
    classes, alpha = split_classes_alpha(x, num_visible)
    idx = torch.argmax(classes, dim=1) + 1        # (N,H,W) in {1..K}
    transparent = (alpha.squeeze(1) < alive_threshold)
    idx = idx.masked_fill(transparent, 0)         # 0 is transparent
    return idx

@torch.no_grad()
def indices_to_state(idx, num_visible: int, alive_threshold: float, device=None, dtype=torch.float32):
    """
    Palette indices -> model state.
      idx: (N,H,W) where 0 = transparent, 1..K = visible classes
    """
    if device is None:
        device = idx.device
    N, H, W = idx.shape

    # Map 1..K -> 0..K-1, clamp 0->0 (we'll zero later)
    clamped = (idx - 1).clamp(min=0, max=num_visible-1)
    oh = F.one_hot(clamped, num_classes=num_visible).permute(0,3,1,2).to(dtype=dtype, device=device)  # (N,K,H,W)

    # Zero out classes where transparent
    opaque = (idx != 0).to(dtype=dtype, device=device).unsqueeze(1)  # (N,1,H,W)
    oh = oh * opaque

    # alpha = 1 for opaque, 0 for transparent (you can set to alive_threshold if you prefer soft start)
    alpha = opaque.clone()

    return torch.cat([oh, alpha], dim=1)  # (N, K+1, H, W)

@torch.no_grad()
def state_to_rgba(x, palette_rgba_u8, num_visible: int, alive_threshold: float, alpha_mode='sigmoid'):
    """
    Use model alpha as output alpha.
    If alpha<threshold: force palette index 0 and alpha=0.
    Else: color by argmax class (1..K).
    """
    device = x.device
    pal = torch.as_tensor(palette_rgba_u8, device=device, dtype=torch.uint8).view(-1, 4)  # (P,4)

    idx = state_to_indices(x, num_visible, alive_threshold)                               # (N,H,W)
    rgb = pal[idx.long()][..., :3].to(torch.float32)/255.0                                # (N,H,W,3)
    rgb = rgb.permute(0,3,1,2).contiguous()                                               # (N,3,H,W)

    _, alpha_raw = split_classes_alpha(x, num_visible)                                    # (N,1,H,W)
    fore = (idx != 0).unsqueeze(1)                                                        # (N,1,H,W) bool
    
    if alpha_mode == 'binary':
        a = fore.float()
    elif alpha_mode == 'sigmoid':
        a = torch.sigmoid(alpha_raw) * fore.float()
    elif alpha_mode == 'palette':
        pal_a = pal[idx.long()][..., 3:4].permute(0,3,1,2).contiguous()
        a = pal_a
    elif alpha_mode == 'palette_mult':
        pal_a = pal[idx.long()][..., 3:4].permute(0,3,1,2).contiguous()
        a = torch.clamp(alpha_raw, 0.0, 1.0) * pal_a
    else: # clamped
        a = torch.clamp(alpha_raw, 0.0, 1.0) * fore.float()

    # rgb = torch.clamp(rgb, 0.0, 1.0)
    # a   = torch.clamp(a,   0.0, 1.0)

    return torch.cat([rgb, a], dim=1)   


def idx_to_oh(c_idx: ColourIdxTensor, num_visible: int) -> ColourOneHotAlphaFirstTensor:
    is_batched = c_idx.ndim == 3
    if not is_batched:
        c_idx = cast(ColourIdxTensor, c_idx.unsqueeze(0))

    # c_idx = c_idx.long().unsqueeze(1)  # (N, 1, H, W)
    # one_hot = F.one_hot(c_idx, num_classes=num_visible) # (N, H, W, C)
    # out = one_hot.squeeze(1).permute(0, 3, 1, 2).float()  # (N, C, H, W)
    c_idx = c_idx.long() # type: ignore
    oh = F.one_hot(c_idx, num_classes=num_visible)  # (N, H, W, C)
    out = oh.permute(0, 3, 1, 2).float()  # (N, C, H, W)

    return ColourOneHotAlphaFirstTensor(out if is_batched else out[0])


def rgba_tensor_to_pil(rgba: torch.Tensor, batch: Optional[int] = 0, premultiply: bool = False) -> Image.Image:
    """
    rgba: (N,4,H,W) or (4,H,W), float in [0,1] or arbitrary → clamps to [0,1]
    batch: which N to pick if batched (None = expect 3D tensor)
    premultiply: multiply RGB by alpha before saving (usually leave False for PNG)
    """
    if rgba.ndim == 4:
        assert batch is not None and 0 <= batch < rgba.size(0)
        rgba = rgba[batch]
    assert rgba.size(0) == 4, f"Expected 4 channels, got {rgba.size(0)}"

    x = rgba.detach().clamp(0, 1).mul(255).add(0.5).to(torch.uint8)      # (4,H,W)
    x = x.permute(1, 2, 0).contiguous().cpu().numpy()                    # (H,W,4)

    if premultiply:
        a = x[..., 3:4].astype(np.uint16)
        rgb = x[..., :3].astype(np.uint16)
        rgb = (rgb * a + 127) // 255
        x = np.concatenate([rgb.astype(np.uint8), a.astype(np.uint8)], axis=-1)

    return Image.fromarray(x, mode="RGBA")


def save_state_as_png(state: torch.Tensor,
                      palette_rgba_u8,
                      num_discrete: int,
                      alive_threshold: float,
                      out_path: Path,
                      batch: int = 0,
                      premultiply: bool = False) -> None:
    """
    state: (N, K+1, H, W) [K classes, 1 alpha]
    """
    rgba = state_to_rgba(state, palette_rgba_u8, num_discrete, alive_threshold)  # (N,4,H,W)
    img = rgba_tensor_to_pil(rgba, batch=batch, premultiply=premultiply)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)  # PNG inferred by suffix

# ------------------------------------------------------------------------------
# Preview/display
# ------------------------------------------------------------------------------

import torch

def print_ascii_grid(t: torch.Tensor,
                     palette: str = " .'`^,:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"):
    """
    Render a tensor as ASCII characters.
    - t: (H,W) or (1,H,W) tensor (float or int). NaNs render as spaces.
    - palette: darkest->brightest characters. Length controls resolution.
               (Provide 12 chars if you want exactly 12 “levels”.)
    """
    # accept (1,H,W)
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 2:
        raise ValueError(f"Expected (H,W) or (1,H,W), got {tuple(t.shape)}")

    # work on CPU float
    x = t.detach().to(torch.float32).cpu()

    # handle NaNs separately
    nan_mask = torch.isnan(x)
    finite = x[~nan_mask]
    if finite.numel() == 0:
        print("\n".join([" " * x.shape[1] for _ in range(x.shape[0])]))
        return

    vmin, vmax = finite.min().item(), finite.max().item()
    if vmax == vmin:  # flat image → use middle of palette
        idx = torch.full_like(x, fill_value=len(palette)//2, dtype=torch.long)
    else:
        xn = (x - vmin) / (vmax - vmin)                 # [0,1]
        idx = (xn * (len(palette) - 1)).round().to(torch.long)

    # put spaces for NaNs
    idx[nan_mask] = -1

    # map indices → chars
    chars = [palette[i] if i >= 0 else ' ' for i in idx.view(-1).tolist()]
    rows = ["".join(chars[r*x.shape[1]:(r+1)*x.shape[1]]) for r in range(x.shape[0])]
    print("\n".join(rows))

def nchw_zoom(c_rgb: T_RGB_A, scale: int = 4) -> T_RGB_A:
    if c_rgb.ndim == 3:
        # CHW -> NCHW
        c_rgb = c_rgb.unsqueeze(0)  # type: ignore
        is_batched = False
    elif c_rgb.ndim == 4:
        is_batched = True
    else:
        raise ValueError(f"Expected 3D or 4D input, got shape {c_rgb.shape}")

    assert c_rgb.shape[1] in (3, 4), f"Expected 3 or 4 channels, got {c_rgb.shape[1]}"

    # Repeat spatial dimensions
    # c_rgb = c_rgb.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3) # type: ignore
    zoomed = F.interpolate(c_rgb, scale_factor=scale, mode="nearest")  # type: ignore

    out = zoomed if is_batched else zoomed[0]  # type: ignore

    return cast(T_RGB_A, out)


def nchw_display(
    c_rgb: T_RGB_A,
    title: Union[str, None] = None,
    figsize: tuple[int, int] = (6, 6),
    checker_size: int = 8,
):
    is_batched = c_rgb.ndim == 4
    if not is_batched:
        c_rgb = c_rgb.unsqueeze(0)  # type: ignore

    c_rgb = c_rgb.permute(0, 2, 3, 1).detach().cpu().numpy()  # type: ignore
    c_rgb = np.clip(c_rgb, 0, 1)  # ensure values are in [0, 1] # type: ignore

    # if 3 channels, add alpha 1.0
    if c_rgb.shape[-1] == 3:
        alpha_channel = np.ones((*c_rgb.shape[:2], 1), dtype=c_rgb.dtype)
        c_rgb = np.concatenate([c_rgb, alpha_channel], axis=-1)


    img = c_rgb[0]  # Take first in batch
    h, w, _ = img.shape

    # --- Create checkerboard background ---
    c0 = 0.85
    c1 = 0.65
    rows = (np.arange(h) // checker_size) % 2
    cols = (np.arange(w) // checker_size) % 2
    checkerboard = np.where((rows[:, None] + cols[None, :]) % 2 == 0, c0, c1)
    checkerboard_rgb = np.dstack([checkerboard]*3)  # (H, W, 3)

    # --- Alpha blend image over checkerboard ---
    alpha = img[..., 3:4]  # (H, W, 1)
    blended = alpha * img[..., :3] + (1 - alpha) * checkerboard_rgb

    # Create matplotlib figure
    plt.figure(figsize=figsize)  # type: ignore
    plt.imshow(blended, interpolation="nearest")  # type: ignore
    plt.axis("off")  # type: ignore
    if title:
        plt.title(title)  # type: ignore
    plt.tight_layout()
    plt.show()  # type: ignore


def tile2d(
    a: np.ndarray[Any, Any], width: Optional[int] = None
) -> np.ndarray[Any, Any]:
    """
    Tile a 2D array into a grid.

    Parameters:
        a: 2D array of shape (n, h, w, c)
        w (int): Width of the grid. If None, it will be set to the square root of n.

    Returns:
        A 2D array of shape (h * grid_height, w * grid_width, c) where
        grid_height and grid_width are determined by the number of images
        and the specified width.
    """
    # a = a.permute(0, 2, 3, 1)
    # a = np.asarray(a)   # convert to numpy array
    if width is None:
        width = int(np.ceil(np.sqrt(len(a))))  # get sqrt to make squarish grid
    th, tw = a.shape[1:3]  # get height and width of each tile
    pad = (width - len(a)) % width  # calculate padding for last row
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), "constant")  # pad with zeroes
    h = len(a) // width  # calculate height of the grid
    a = a.reshape([h, width] + list(a.shape[1:]))  # reshape to grid [h, w, th, tw]
    # move axis 2 to position 1 (grid_height, th, grid_width, tw)
    # then reshape to (grid_height * th, grid_width * tw, ...)
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * width] + list(a.shape[4:]))
    return a


def export_model(ca: torch.nn.Module, base_fn: str):
    torch.save(ca.state_dict(), base_fn + ".pth") # type: ignore


def plot_loss(loss_log: list[float]):
    plt.figure(figsize=(10, 4)) # type: ignore
    plt.title("Loss history (log10)")  # type: ignore
    plt.plot(np.log10(loss_log), ".", alpha=0.1)  # type: ignore
    plt.show()  # type: ignore


def save_coordinate_rgb_image(width: int, height: int, filename: str):
    # Create normalized coordinate grid (0 to 1)
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)

    # Scale to 0–127 (50% of 255) + 63
    r = (grid_x * 127 + 63).astype(np.uint8)  # X to Red
    g = (grid_y * 127 + 63).astype(np.uint8)  # Y to Green
    b = np.zeros_like(r, dtype=np.uint8)  # Blue = 0

    # Stack into RGB image
    rgb = np.stack([r, g, b], axis=-1)

    # Save using PIL
    img = PIL.Image.fromarray(rgb, mode="RGB")
    img.save(filename)
    print(f"Saved coordinate-colored image to {filename}")


def frame_image(image: PIL.Image.Image, expand: int, text: str) -> PIL.Image.Image:
    """
    Process a PIL image with the following steps:
    1. Expand by initial_expand (top, left, right, bottom).
    2. Add a 1px black border.
    3. Expand by final_expand (top, left, right, bottom).
    4. Add text at the bottom (with automatic bottom padding if necessary).

    :param image: PIL.Image object
    :param initial_expand: Tuple of (top, left, right, bottom) for first expansion
    :param final_expand: Tuple of (top, left, right, bottom) for second expansion
    :param text: String, may contain newline characters
    :return: Modified PIL.Image object
    """
    # Step 1: Initial expansion
    image = ImageOps.expand(image, border=expand, fill="white")

    # Step 4: Add text at the bottom
    draw = ImageDraw.Draw(image)

    # Use a small default font
    try:
        font = ImageFont.truetype("Helvetica.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    # Split text into lines and calculate total height
    text_lines = text.split("\n")
    line_spacing = 2

    # Calculate the total height of the text block
    line_height: int = 0
    max_width: int = 0
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)  # get bounding box of the text
        line_height += int(
            bbox[3] - bbox[1] + line_spacing
        )  # bbox[3] is the bottom y, bbox[1] is the top y
        max_width = max(
            max_width, int(bbox[2]) - int(bbox[0])
        )  # bbox[2] is right x, bbox[0] is left x

    # Expand canvas at bottom if needed
    if image.width < max_width:
        image = ImageOps.expand(
            image, border=(0, 0, max_width - image.width, 0), fill="white"
        )

    # Create a new image with expanded bottom area for text
    new_image = Image.new(
        "RGBA", (image.width, image.height + line_height + expand), "white"
    )
    new_image.paste(image, (0, 0))

    # Draw text at bottom
    draw = ImageDraw.Draw(new_image)
    y = image.height

    for line in text_lines:
        draw.text((expand, y), line, fill="black", font=font)
        y += (
            draw.textbbox((0, 0), line, font=font)[3]
            - draw.textbbox((0, 0), line, font=font)[1]
            + line_spacing
        )

    return new_image


def load_palette_from_image(path: Path) -> list[int]:
    """
    Load a palettized (P-mode) image and return its palette as a flat list of length 256*4 (RGBA uint8).
    If the image has no palette, raises an error.
    """
    img = PIL.Image.open(path)
    if img.mode != "P":
        print(f"Image is not in palettized (P) mode: {path}")
        img = img.convert("P")

    # Expand the palette to RGBA (length <= 256*4)
    palette = img.getpalette(rawmode="RGBA")
    if palette is None:
        raise ValueError(f"No palette found in image: {path}")

    entries = 256 * 4
    pal256 = palette[:entries] + [0] * (entries - len(palette))
    return pal256


def load_indexed_target_image(
    path: Path,
    padding: int,
    num_visible: int,
    alive_threshold: float = 0.1,
    device = None,
):
    """
    Load a palettized (P-mode) image where:
      - index 0 is fully transparent
      - indices 1..K are the K visible classes

    Returns:
      x:  (1, K+1, H+2p, W+2p) float32  [K classes, 1 alpha]
      pal256: flat list of length 256*4 (RGBA uint8)
    """
    img = PIL.Image.open(path)
    if img.mode != "P":
        raise ValueError(f"Image is not in palettized (P) mode: {path}, got mode {img.mode}")

    # Expand the palette to RGBA (length <= 256*4)
    palette = img.getpalette(rawmode="RGBA")
    if palette is None:
        raise ValueError(f"No palette found in image: {path}")
    
    entries = 256 * 4
    pal256 = palette[:entries] + [0] * (entries - len(palette))
    
    trans = img.info.get("transparency")
    if isinstance(trans, bytes):         # per-index alpha (PNG tRNS)
        for i, a in enumerate(trans):
            if i < 256:
                pal256[i*4 + 3] = a
    elif isinstance(trans, list):        # sometimes a list
        for i, a in enumerate(trans):
            if i < 256:
                pal256[i*4 + 3] = a
    elif isinstance(trans, int):         # single transparent index (e.g. GIF)
        pal256[trans*4 + 3] = 0

    pal256[3] = 0  # ensure index 0 is fully transparent

    indices = torch.from_numpy(np.array(img, dtype=np.uint8)).to(torch.long)

    unused = indices > num_visible
    if unused.any():
        indices = indices.masked_fill(unused, 0)

    # convert to CA state (but no hidden channels)
    x = indices_to_state(
        idx=indices.unsqueeze(0),              # (1, H, W)
        num_visible=num_visible,
        alive_threshold=alive_threshold,
        device=device,
    )                                          # (1, K+1, H, W)

    # Pad (N,C,H,W) with (left,right,top,bottom)
    x = F.pad(x, (padding, padding, padding, padding))

    return x, pal256

import yaml
from dataclasses import asdict

def save_config(cfg, output_dir, filename="config.yaml"):
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert dataclass to dict, then dump as YAML
    with open(output_path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)