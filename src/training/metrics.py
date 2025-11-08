import torch
from typing import Optional, Sequence, Tuple, Union

# metrics
# ------------------------------------------------------------------------------

def mse_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred - target) ** 2).mean().item()

def exact_rgb_accuracy(
    pred_rgb: torch.Tensor,          # (N, 3[, or 4], H, W) float
    target_rgb: torch.Tensor,        # same shape as pred_rgb
    use_uint8_compare: bool = False, # set True if you want “visually exact” (0–255) equality
    mask: Optional[torch.Tensor] = None # (N,1,H,W) or (N,H,W) boolean/0-1 to include only some pixels
) -> float:
    """
    Returns fraction of pixels whose ALL visible channels match exactly.
    If use_uint8_compare=True, compares after rounding to uint8 [0..255].
    """
    assert pred_rgb.shape == target_rgb.shape, "pred and target must have same shape"
    C = pred_rgb.size(1)

    if use_uint8_compare:
        p = (pred_rgb.clamp(0,1) * 255.0).round().to(torch.uint8)
        t = (target_rgb.clamp(0,1) * 255.0).round().to(torch.uint8)
    else:
        p, t = pred_rgb, target_rgb

    equal_all = (p == t).all(dim=1)  # (N,H,W)

    if mask is not None:
        m = mask
        if m.dim() == 4 and m.size(1) == 1: m = m[:,0]   # (N,H,W)
        m = m.bool()
        num = equal_all[m].sum()
        den = m.sum().clamp_min(1)
    else:
        num = equal_all.sum()
        den = equal_all.numel()

    return (num.float() / float(den)).item()


def exact_index_accuracy(
    pred_idx: torch.Tensor,      # (N,H,W) long/int
    target_idx: torch.Tensor,    # (N,H,W) long/int
    mask: Optional[torch.Tensor] = None # (N,H,W) or (N,1,H,W)
) -> float:
    """Zero-tolerance accuracy for palette/class index images."""
    assert pred_idx.shape == target_idx.shape
    equal_px = (pred_idx == target_idx)
    if mask is not None:
        m = mask
        if m.dim() == 4 and m.size(1) == 1: m = m[:,0]
        m = m.bool()
        num = equal_px[m].sum()
        den = m.sum().clamp_min(1)
    else:
        num = equal_px.sum()
        den = equal_px.numel()
    return (num.float() / den.float()).item()

# --- 2) Palette-snap then zero-tolerance ---

def snap_rgb_to_palette(
    pred_rgb: torch.Tensor,                        # (N,3,H,W) or (N,4,H,W), float
    palette: Union[Sequence[float], torch.Tensor], # flat RGBA [r,g,b,a,...] or (K,3|4)
    drop_transparent: bool = False,                # optionally ignore entries with alpha==0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each pixel in pred_rgb, pick the nearest color from `palette`.
    `palette` can be a flat RGBA list [r0,g0,b0,a0, r1,g1,b1,a1, ...] (0..255)
    OR a (K,3|4) tensor/list in either 0..255 or 0..1.

    Returns:
      snapped_rgb : (N,C,H,W) in [0,1]
      snapped_idx : (N,H,W) long indices into the ORIGINAL palette order
    """
    assert pred_rgb.dim() == 4 and pred_rgb.size(1) in (3, 4), "pred must be (N,3|4,H,W)"
    N, C, H, W = pred_rgb.shape
    device = pred_rgb.device

    # ---- Parse palette input (accept flat RGBA or (K,3|4)) ----
    pal = torch.as_tensor(palette, device=device)
    if pal.dim() == 1:
        # flat RGBA -> (K,4)
        if pal.numel() % 4 != 0:
            raise ValueError("Flat palette length must be multiple of 4 (RGBA).")
        pal = pal.view(-1, 4)
    elif pal.dim() == 2:
        if pal.size(1) not in (3, 4):
            raise ValueError("Palette second dim must be 3 (RGB) or 4 (RGBA).")
    else:
        raise ValueError("Palette must be 1D flat RGBA or 2D (K,3|4).")

    # Normalise to float [0,1] if values look like 0..255
    pal = pal.to(torch.float32)
    if pal.max() > 1.001:
        pal = pal / 255.0

    # Optionally drop transparent entries (alpha==0) but preserve original indexing
    if pal.size(1) == 4:
        if drop_transparent:
            keep = pal[:, 3] > 0.0
            kept_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            pal_use = pal[keep][:, :C]  # match image channels
        else:
            kept_idx = torch.arange(pal.size(0), device=device)
            pal_use = pal[:, :C]
    else:
        kept_idx = torch.arange(pal.size(0), device=device)
        pal_use = pal[:, :C]  # (K, C)

    if pal_use.numel() == 0:
        raise ValueError("Palette has no usable entries after filtering.")

    # ---- Nearest colour snapping ----
    # Clamp image into [0,1] to match palette scale
    P = pred_rgb.clamp(0, 1).permute(0, 2, 3, 1).reshape(N, H * W, C)  # (N,HW,C)
    # distances: (N,HW,K)
    d2 = ((P.unsqueeze(2) - pal_use.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=3)
    j = d2.argmin(dim=2)                    # (N,HW) indices into pal_use
    snapped = pal_use[j]                    # (N,HW,C)
    snapped = snapped.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)

    # Map back to original palette indices (important if we dropped transparent)
    snapped_idx = kept_idx[j].view(N, H, W).long()

    return snapped, snapped_idx

def palette_snapped_exact_rgb_accuracy(
    pred_rgb: torch.Tensor,        # (N,3[,4],H,W)
    target_rgb: torch.Tensor,      # (N,3[,4],H,W)
    palette_rgb: torch.Tensor,     # (K,3[,4])
    mask: Optional[torch.Tensor] = None,
    use_uint8_compare: bool = False
) -> float:
    """
    Snap pred_rgb to nearest color in *target's palette*, then exact match vs target.
    """
    snapped, _ = snap_rgb_to_palette(pred_rgb, palette_rgb)
    return exact_rgb_accuracy(snapped, target_rgb, use_uint8_compare=use_uint8_compare, mask=mask)


def palette_snapped_exact_index_accuracy(
    pred_rgb: torch.Tensor,        # (N,3[,4],H,W)
    target_idx: torch.Tensor,      # (N,H,W) long
    palette_rgb: torch.Tensor      # (K,3[,4])
) -> float:
    """
    Snap pred_rgb to nearest palette entry, then compare indices to target_idx.
    """
    _, snapped_idx = snap_rgb_to_palette(pred_rgb, palette_rgb)
    return exact_index_accuracy(snapped_idx.long(), target_idx.long())