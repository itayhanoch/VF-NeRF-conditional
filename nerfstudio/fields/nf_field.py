"""Conditional Normalizing Flow field.

Models P((3D position, 3D direction) | condition) via a conditional RealNVP, so that a
condition vector (a DINOv2 feature taken at some source-image pixel) can be sampled into
plausible novel-view ray candidates. The flow itself is ported from
itayhanoch/conditional-normalizing-flows-toy -- see nerfstudio/field_components/condflow.py.
"""
from typing import Optional

import torch
from torch import nn

from nerfstudio.field_components.condflow import get_generator


class DimReducerMLP(nn.Module):
    """Reduces `in_dim` to `out_dim`, dividing by `divide_factor` at each hidden
    layer until within one division of `out_dim`, then a final Linear lands
    exactly on `out_dim`. E.g. in_dim=384, out_dim=6, divide_factor=8 -> layer
    sizes [384, 48, 6] (two Linear layers, one ReLU between)."""

    def __init__(self, in_dim: int, out_dim: int, divide_factor: int = 8):
        super().__init__()
        dims = [in_dim]
        while dims[-1] // divide_factor > out_dim:
            dims.append(dims[-1] // divide_factor)
        dims.append(out_dim)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalNFField(nn.Module):
    """Conditional RealNVP over 6D (position, direction) vectors, conditioned on a
    context vector (e.g. a DINOv2 feature at the source pixel a ray passes through).

    Args:
        context_dim: dimensionality of the condition vector (384 for dinov2_vits14).
        num_dims: dimensionality of the modeled vector (3D position + 3D direction = 6).
        num_blocks: number of coupling+batchnorm blocks (toy repo's real-data
            `dual_cond_rgb` variant uses 8 for a 2D/3-cond-dim problem).
        hidden_dim: hidden width of each coupling layer's scale/translate MLPs.
        cond_prior: also condition the base distribution's mean/log-var on `context`
            (the toy repo's best-performing "dual_cond" variant).
        use_cond_in_coupling: condition each coupling layer's scale/translate nets.
        use_batchnorm: interleave BatchNormFlow layers between couplings.
        reduce_dim: if set, `context` is passed through a `DimReducerMLP` (trained
            jointly with the flow) that reduces it from `context_dim` to
            `reduce_dim` before it ever reaches the flow. Left as `None` (the
            default), `context` reaches the flow unreduced -- the exact prior
            behavior, so existing checkpoints keep loading unchanged.
        reduce_divide_factor: division factor used by the `DimReducerMLP` when
            `reduce_dim` is set (see `DimReducerMLP`).
    """

    def __init__(
        self,
        context_dim: int,
        num_dims: int = 6,
        num_blocks: int = 8,
        hidden_dim: int = 128,
        cond_prior: bool = True,
        use_cond_in_coupling: bool = True,
        use_batchnorm: bool = True,
        reduce_dim: Optional[int] = None,
        reduce_divide_factor: int = 8,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.num_dims = num_dims
        self.reduce_dim = reduce_dim
        if reduce_dim is None:
            self.reduce = nn.Identity()
            cond_dim = context_dim
        else:
            self.reduce = DimReducerMLP(context_dim, reduce_dim, divide_factor=reduce_divide_factor).to(device)
            cond_dim = reduce_dim
        self.flow = get_generator(
            num_inputs=num_dims,
            num_cond_inputs=cond_dim,
            device=device,
            num_blocks=num_blocks,
            num_hidden=hidden_dim,
            cond_prior=cond_prior,
            use_cond_in_coupling=use_cond_in_coupling,
            use_batchnorm=use_batchnorm,
        )

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x: [B, num_dims] (position, direction). context: [B, context_dim],
        optionally reduced to [B, reduce_dim] by `self.reduce` first (a no-op
        when `reduce_dim` was not set). Returns [B, 1] log-density."""
        return self.flow.log_prob(x, self.reduce(context))

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        """Draw `num_samples` (position, direction) points conditioned on `context`.

        `context` may be a single [context_dim] vector (broadcast to every sample --
        the interactive-UI "click one point -> sample N" case) or an already-batched
        [num_samples, context_dim] tensor (the training case, one condition per ray).
        Reduced by `self.reduce` first (a no-op when `reduce_dim` was not set).

        Returns [num_samples, num_dims]. Unlike normflows' `.sample()`, this does NOT
        also return a log-probability -- call `log_prob(samples, context)` separately
        to rank/score the drawn samples (matches the ported flow's own
        `FlowSequential.sample` signature).
        """
        context = self.reduce(context)
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(num_samples, -1)
        return self.flow.sample(num_samples=num_samples, cond_inputs=context)
