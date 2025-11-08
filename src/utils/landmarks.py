import math
import torch

from src.config.run import RunCfg

def make_landmark_fn(config: RunCfg):
    lm_type = config.model.landmarks.type
    params = config.model.landmarks.params
    H = config.target.target_size + 2 * config.target.target_padding
    W = config.target.target_size + 2 * config.target.target_padding
    print(f"Making landmarks of type '{lm_type}' with params {params} and size ({H}, {W})")
    device = config.compute.device

    fns = {
        "coordinate": coordinate_grid,
        "dots": grid_dots,
        "checkerboard": checkerboard,
        "gaussian": gaussian_grid,
        "posenc_sincos_2d": posenc_sincos_2d,
        "posenc_fourier_features": posenc_fourier_features,
    }

    if lm_type is None or lm_type == "" or lm_type.lower() == "none":
        return None

    if lm_type not in fns:
        raise ValueError(f"Unknown landmarks type: {lm_type}")

    lm_fn = fns[lm_type]
    return lm_fn(H=H, W=W, device=device, **params)


def coordinate_grid(H, W, device=None, normalise=True):
    """
    Returns 2 channels (y,x) with either pixel indices or [-1,1] range.
    Shape: (2, H, W)
    """
    ys = torch.linspace(0, H-1, H, device=device)
    xs = torch.linspace(0, W-1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    if normalise:
        yy = 2.0 * (yy / (H-1)) - 1.0
        xx = 2.0 * (xx / (W-1)) - 1.0
    return torch.stack([yy, xx], dim=0)

def grid_dots(H, W, spacing=5, device=None, value=1.0):
    """
    Binary mask with a dot every `spacing` pixels on both axes.
    Shape: (1, H, W)
    """
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    dot_y = (ys % spacing == 0).float().unsqueeze(1).expand(H, W)
    dot_x = (xs % spacing == 0).float().unsqueeze(0).expand(H, W)
    dots = (dot_y * dot_x) * value
    return dots.unsqueeze(0)

def checkerboard(H, W, spacing=5, device=None, low=0.0, high=1.0):
    """
    Checker pattern with block size `spacing`.
    Shape: (1, H, W)
    """
    ys = torch.arange(H, device=device) // spacing
    xs = torch.arange(W, device=device) // spacing
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    chk = ((yy + xx) % 2).float()
    return (low + (high - low) * chk).unsqueeze(0)

def gaussian_grid(H, W, spacing=16, sigma=2.0, device=None):
    """
    Gaussian bumps centered on every lattice point separated by `spacing`.
    Shape: (1, H, W)
    """
    # Distance to nearest lattice point using modulo (tile-aware distance)
    ys = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
    xs = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
    dy = (ys + spacing//2) % spacing - spacing//2
    dx = (xs + spacing//2) % spacing - spacing//2
    d2 = (dx**2 + dy**2).float()
    g = torch.exp(-0.5 * d2 / (sigma**2))
    return g.unsqueeze(0)

def posenc_sincos_2d(H, W, C, device=None, min_freq=1.0):
    """
    2D sinusoidal positional encoding (transformer-style).
    C must be divisible by 4. Returns (C, H, W) with [sin_x, cos_x, sin_y, cos_y] groups.
    Frequencies are log-spaced from min_freq up to ~W/2 and ~H/2.
    """
    assert C % 4 == 0, "C must be divisible by 4."
    c_per_axis = C // 2
    c_each = c_per_axis // 2

    # Coordinates in [0,1]
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    # Log-spaced frequencies
    x_max = max(1, W//2)
    y_max = max(1, H//2)
    fxs = torch.logspace(math.log10(min_freq), math.log10(x_max), c_each, device=device)
    fys = torch.logspace(math.log10(min_freq), math.log10(y_max), c_each, device=device)

    # Build channels
    enc_x = []
    for f in fxs:
        enc_x.append(torch.sin(2*math.pi * f * xx))
        enc_x.append(torch.cos(2*math.pi * f * xx))

    enc_y = []
    for f in fys:
        enc_y.append(torch.sin(2*math.pi * f * yy))
        enc_y.append(torch.cos(2*math.pi * f * yy))

    enc = torch.stack(enc_x + enc_y, dim=0)  # (C, H, W) with C = 2*c_each + 2*c_each = 4*c_each
    # If C is bigger (due to even divisibility), pad or trim (here we trim/pad zeros)
    if enc.shape[0] < C:
        pad = torch.zeros(C - enc.shape[0], H, W, device=device)
        enc = torch.cat([enc, pad], dim=0)
    elif enc.shape[0] > C:
        enc = enc[:C]
    return enc

def posenc_fourier_features(H, W, C, device=None, scale=10.0, seed=None):
    """
    Tancik et al.-style random Fourier features on (x,y) in [-1,1].
    Returns (C, H, W). C should be even; channels are [sin(Bx), cos(Bx)] concatenated.
    """
    assert C % 2 == 0, "C should be even."
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    # Normalized coords in [-1,1]
    yy, xx = coordinate_grid(H, W, device=device, normalize=True)
    coords = torch.stack([xx, yy], dim=0).reshape(2, -1)  # (2, H*W)

    # Random projection matrix B ~ N(0, scale^2)
    d = C // 2
    B = torch.normal(0, scale, size=(d, 2), generator=rng, device=device)  # (d,2)
    proj = B @ coords  # (d, H*W)
    sin = torch.sin(proj).reshape(d, H, W)
    cos = torch.cos(proj).reshape(d, H, W)
    return torch.cat([sin, cos], dim=0)  # (C, H, W)