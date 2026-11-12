from __future__ import annotations

import torch

from fasttomo.algorithms._multigpu import _RankContext, _slab_tv
from fasttomo.regularizers import tv_loss_3d


def test_slab_tv_single_row_matches_tv_loss_without_division() -> None:
    vol = torch.randn(1, 9, 7, dtype=torch.float32, requires_grad=True)
    ctx = _RankContext(rank=0, n_ranks=1, rows=(0, 1, 1), halo=lambda _: (None, None))

    owned, hidden = _slab_tv(ctx, vol, tv_loss_3d)

    assert hidden is None
    assert torch.equal(owned, tv_loss_3d(vol))

    owned.backward()
    torch.testing.assert_close(vol.grad, torch.zeros_like(vol), rtol=0, atol=0)
