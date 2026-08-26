"""Conditional Normalizing Flow field.

Models P((3D position, 3D direction) | condition) via a conditional RealNVP, so that a
condition vector (a DINOv2 feature taken at some source-image pixel) can be sampled into
plausible novel-view ray candidates. The flow itself is ported from
itayhanoch/conditional-normalizing-flows-toy -- see nerfstudio/field_components/condflow.py.
"""
import torch
from torch import nn

from nerfstudio.field_components.condflow import get_generator


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
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.num_dims = num_dims
        self.flow = get_generator(
            num_inputs=num_dims,
            num_cond_inputs=context_dim,
            device=device,
            num_blocks=num_blocks,
            num_hidden=hidden_dim,
            cond_prior=cond_prior,
            use_cond_in_coupling=use_cond_in_coupling,
            use_batchnorm=use_batchnorm,
        )

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x: [B, num_dims] (position, direction). context: [B, context_dim].
        Returns [B, 1] log-density."""
        return self.flow.log_prob(x, context)

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        """Draw `num_samples` (position, direction) points conditioned on `context`.

        `context` may be a single [context_dim] vector (broadcast to every sample --
        the interactive-UI "click one point -> sample N" case) or an already-batched
        [num_samples, context_dim] tensor (the training case, one condition per ray).

        Returns [num_samples, num_dims]. Unlike normflows' `.sample()`, this does NOT
        also return a log-probability -- call `log_prob(samples, context)` separately
        to rank/score the drawn samples (matches the ported flow's own
        `FlowSequential.sample` signature).
        """
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(num_samples, -1)
        return self.flow.sample(num_samples=num_samples, cond_inputs=context)
