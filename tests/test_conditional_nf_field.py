"""Sanity tests for ConditionalNFField / nerfstudio.field_components.condflow,
mirroring itayhanoch/conditional-normalizing-flows-toy's own tests/test_layers.py
(invertibility, shapes) plus an isolation test specific to this project: verifying
the conditioning mechanism actually discriminates between conditions BEFORE ever
touching the real NeRF/DINO/ray training pipeline. Run manually (no CI configured
in this repo): `pytest tests/`, e.g. whenever condflow.py/nf_field.py changes, or
before committing to a real Colab training run.
"""
import torch

from nerfstudio.fields.nf_field import ConditionalNFField


def _make_field(context_dim, **kwargs):
    defaults = dict(num_dims=6, num_blocks=2, hidden_dim=16, cond_prior=False, use_cond_in_coupling=True, use_batchnorm=True, device="cpu")
    defaults.update(kwargs)
    return ConditionalNFField(context_dim=context_dim, **defaults)


def test_forward_inverse_roundtrip():
    field = _make_field(context_dim=8)
    for mode in ("train", "eval"):
        getattr(field, mode)()  # BatchNormFlow behaves differently train vs eval -- check both
        x = torch.randn(16, 6)
        cond = torch.randn(16, 8)
        z, _ = field.flow(x, cond, mode="direct")
        x_hat, _ = field.flow(z, cond, mode="inverse")
        assert torch.allclose(x, x_hat, atol=1e-4), f"roundtrip failed in {mode} mode"


def test_log_prob_and_sample_shapes():
    field = _make_field(context_dim=8, cond_prior=True)
    field.eval()

    x = torch.randn(10, 6)
    cond = torch.randn(10, 8)
    log_p = field.log_prob(x, cond)
    assert log_p.shape == (10, 1)

    # batched-context sampling (the training-time shape)
    samples = field.sample(num_samples=10, context=cond)
    assert samples.shape == (10, 6)

    # single-vector-context sampling (the interactive-UI "click one point" shape)
    samples_single = field.sample(num_samples=100, context=cond[0])
    assert samples_single.shape == (100, 6)


def test_condition_discriminates_two_clusters():
    """Overfit on 2 fixed condition vectors, each paired with a distinct,
    well-separated 6D Gaussian; verify (a) log_prob favors the matching
    condition/target pairing over the mismatched one, and (b) sample() draws
    points near the correct cluster with roughly the right spread. Isolates the
    conditioning mechanism (condflow.CouplingLayer's context concatenation, plus
    the conditional prior) from the NeRF/DINO/ray machinery entirely -- the
    cheapest place to catch a "context isn't actually reaching the model" bug.

    Recipe (lr, schedule, step count, grad clipping) is empirically validated:
    an earlier attempt at lr=1e-2/300 steps/no clipping did not converge.
    """
    torch.manual_seed(0)
    context_dim = 4
    cond_a = torch.zeros(context_dim)
    cond_a[0] = 1.0
    cond_b = torch.zeros(context_dim)
    cond_b[1] = 1.0
    mean_a = torch.full((6,), -5.0)
    mean_b = torch.full((6,), 5.0)
    data_std = 0.1

    field = _make_field(context_dim=context_dim, num_blocks=8, hidden_dim=128, cond_prior=True)
    optimizer = torch.optim.Adam(field.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)

    n = 64
    field.train()
    for _ in range(1500):
        x_a = mean_a + data_std * torch.randn(n, 6)
        x_b = mean_b + data_std * torch.randn(n, 6)
        cond_batch = torch.cat([cond_a.unsqueeze(0).expand(n, -1), cond_b.unsqueeze(0).expand(n, -1)], dim=0)
        x = torch.cat([x_a, x_b], dim=0)

        loss = -field.log_prob(x, cond_batch).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

    field.eval()
    with torch.no_grad():
        log_p_match = field.log_prob(mean_a.unsqueeze(0), cond_a.unsqueeze(0))
        log_p_mismatch = field.log_prob(mean_a.unsqueeze(0), cond_b.unsqueeze(0))
        assert log_p_match.item() > log_p_mismatch.item(), "matching condition should score higher than a mismatched one"

        samples = field.sample(num_samples=200, context=cond_a)
        assert (samples.mean(dim=0) - mean_a).abs().max() < 1.0, "samples under cond_a should cluster near mean_a"
        assert (samples.std(dim=0) - data_std).abs().max() < 0.3, "sample spread should roughly match the true generating std"
