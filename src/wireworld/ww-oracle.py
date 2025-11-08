# wireworld oracle
# accepts a wireworld state and output the next step
import torch

from src.utils import T_CA, load_indexed_target_image, nchw_display, nchw_zoom
from src.utils import state_to_rgba

def wireworld_step(state: T_CA) -> T_CA:
    # channel order is [wire, tail, head, empty]
    wire = state[:, 0, :, :]
    tail = state[:, 1, :, :]
    head = state[:, 2, :, :]
    empty = state[:, 3, :, :]

    # wire -> head rule
    # find number of neighbours that are head cells
    headcount = torch.nn.functional.conv2d(head,
                                           weight=torch.ones((1, 1, 3, 3), device=head.device),
                                           stride=1,
                                           padding=1)
    # mask where headcount = 1 or 2
    head_mask = (headcount == 1) | (headcount == 2)

    # next states
    next_head = head_mask & (wire != 0)
    next_wire = ((wire != 0) | (tail != 0)) & ~next_head
    next_wire = next_wire.float()
    next_tail = head.clone()
    next_empty = empty.clone()

    # combine back into (N, C, H, W)
    next_state = torch.stack([next_wire, next_tail, next_head, next_empty], dim=1)

    return next_state

# if main
if __name__ == "__main__":
    state, palette = load_indexed_target_image(
        "data/targets/wireworld/golly-not-32.png",
        padding=0,
        num_visible=4
    )

    for _ in range(30):
        state = wireworld_step(state)
        # add an extra alpha channel, with ones where any of the other channels have non-zero
        state_alpha = torch.cat([state, (state != 0).any(dim=1, keepdim=True).float()], dim=1)
        # zoom and display the state
        rgba = state_to_rgba(state_alpha, palette, num_visible=4, alive_threshold=0.1)
        nchw_display(nchw_zoom(rgba, 2), fmt="png")