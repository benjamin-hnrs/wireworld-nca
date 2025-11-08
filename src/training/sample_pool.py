import torch

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, device='cpu', **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = len(next(iter(slots.values())))
        self._device = device
        for k, v in slots.items():
            assert len(v) == self._size
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.detach().clone().to(device))
            else:
                setattr(self, k, torch.tensor(v, device=device))

    def sample(self, n):
        idx = torch.randint(0, self._size, (n,), device=self._device)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx, device=self._device)
        return batch

    def commit(self, parent, parent_idx):
        with torch.no_grad():
            for k in self._slot_names:
                # original
                getattr(parent, k)[parent_idx] = getattr(self, k).detach()
