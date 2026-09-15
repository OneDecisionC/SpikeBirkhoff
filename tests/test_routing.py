# Copyright (c) 2026 Zhiqi Cai
# SPDX-License-Identifier: MIT
"""Constraint, initialization, and stream-coordinate regression tests."""

import pytest
import torch

from spikebirkhoff.routing import MultiStreamHyperConnection


MODES = MultiStreamHyperConnection.MODES
DS_MODES = ("birkhoff", "ds_projected", "fixed_A0", "fixed_I", "fixed_J")


def inputs():
    generator = torch.Generator().manual_seed(7)
    streams = torch.randn(2, 2, 2, 3, 2, 2, generator=generator, dtype=torch.float64)
    delta = torch.randn(2, 2, 3, 2, 2, generator=generator, dtype=torch.float64)
    return streams, delta


@pytest.mark.parametrize("mode", MODES)
def test_router_initialization_preserves_rng_and_archived_raw_shapes(mode):
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    router = MultiStreamHyperConnection(2, mode, init_scale=0.01)
    assert torch.equal(before, torch.get_rng_state())
    shape = (2,) if mode in ("birkhoff", "fixed_I", "fixed_J") else (2, 2)
    assert tuple(router.mixing_logits.shape) == shape
    assert set(router.state_dict()) == {"read_logits", "write_logits", "mixing_logits"}
    torch.testing.assert_close(router.read_weights, torch.tensor([0.5, 0.5]))
    assert router.write_weights[0] > router.write_weights[1]
    torch.testing.assert_close(router.write_weights.sum(), torch.tensor(2.0))
    assert router.mixing_logits.requires_grad == (mode not in ("fixed_A0", "fixed_I", "fixed_J"))


@pytest.mark.parametrize("mode", ("unconstrained", "row_projected", "ds_projected", "fixed_A0"))
def test_effective_initial_matrix_matches_birkhoff(mode):
    reference = MultiStreamHyperConnection(2, "birkhoff", init_scale=0.01)
    actual = MultiStreamHyperConnection(2, mode, init_scale=0.01)
    torch.testing.assert_close(actual.mixing_matrix(), reference.mixing_matrix(), rtol=0, atol=6e-8)


@pytest.mark.parametrize("mode", ("fixed_A0", "fixed_I"))
@pytest.mark.parametrize("source_form, target_form", (("logits", "matrix"), ("matrix", "logits")))
def test_strict_loading_preserves_both_archived_frozen_parameter_forms(mode, source_form, target_form):
    source = MultiStreamHyperConnection(2, mode, parameterization=source_form)
    target = MultiStreamHyperConnection(2, mode, parameterization=target_form)
    target.load_state_dict(source.state_dict(), strict=True)
    assert target.mixing_logits.shape == source.mixing_logits.shape
    assert not target.mixing_logits.requires_grad
    torch.testing.assert_close(target.mixing_logits, source.mixing_logits, rtol=0, atol=0)
    torch.testing.assert_close(target.mixing_matrix(), source.mixing_matrix(), rtol=0, atol=0)


@pytest.mark.parametrize("mode", DS_MODES)
def test_doubly_stochastic_constraints_and_mean_residual_law(mode):
    router = MultiStreamHyperConnection(2, mode).double()
    if router.mixing_logits.requires_grad:
        with torch.no_grad():
            router.mixing_logits.copy_(torch.linspace(-0.3, 0.8, router.mixing_logits.numel()).reshape_as(router.mixing_logits))
    matrix = router.mixing_matrix()
    torch.testing.assert_close(matrix.sum(0), torch.ones(2, dtype=matrix.dtype), atol=1e-7, rtol=0)
    torch.testing.assert_close(matrix.sum(1), torch.ones(2, dtype=matrix.dtype), atol=1e-7, rtol=0)
    assert matrix.min() >= 0
    streams, delta = inputs()
    torch.testing.assert_close(router(streams, delta).mean(0), streams.mean(0) + delta, atol=2e-7, rtol=1e-7)


def test_mean_contrast_and_nonuniform_read_equations():
    router = MultiStreamHyperConnection(2, "birkhoff").double()
    with torch.no_grad():
        router.mixing_logits.copy_(torch.tensor([-0.4, 0.7]))
        router.read_logits.copy_(torch.tensor([-0.3, 0.6]))
        router.write_logits.copy_(torch.tensor([0.2, -0.5]))
    streams, delta = inputs()
    mean = streams.mean(0)
    contrast = (streams[0] - streams[1]) / 2
    output = router(streams, delta)
    alpha = router.mixing_matrix()[0, 0]
    p, q = router.read_weights, router.write_weights
    torch.testing.assert_close(router.read(streams), mean + (p[0] - p[1]) * contrast)
    torch.testing.assert_close(output.mean(0), mean + delta)
    torch.testing.assert_close((output[0] - output[1]) / 2, (2 * alpha - 1) * contrast + (q[0] - q[1]) * delta / 2)


def test_row_projection_enforces_rows_but_can_leak_contrast_into_mean():
    router = MultiStreamHyperConnection(2, "row_projected").double()
    with torch.no_grad():
        router.mixing_logits.copy_(torch.tensor([[0.9, 0.1], [0.6, 0.4]]))
    matrix = router.mixing_matrix()
    torch.testing.assert_close(matrix, torch.tensor([[0.9, 0.1], [0.6, 0.4]], dtype=torch.float64))
    torch.testing.assert_close(matrix.sum(1), torch.ones(2, dtype=torch.float64))
    assert not torch.allclose(matrix.sum(0), torch.ones(2, dtype=torch.float64))
    streams, delta = inputs()
    assert not torch.allclose(router(streams, delta).mean(0), streams.mean(0) + delta)


@pytest.mark.parametrize("mode, expected", [
    ("row_projected", [[1.0, 0.0], [0.0, 1.0]]),
    ("ds_projected", [[1.0, 0.0], [0.0, 1.0]]),
])
def test_projected_controls_clip_to_boundary(mode, expected):
    router = MultiStreamHyperConnection(2, mode).double()
    with torch.no_grad():
        router.mixing_logits.copy_(torch.tensor([[3.0, -2.0], [-4.0, 5.0]]))
    torch.testing.assert_close(router.mixing_matrix(), torch.tensor(expected, dtype=torch.float64))


def test_unconstrained_mixer_can_amplify_and_violate_conservation():
    router = MultiStreamHyperConnection(2, "unconstrained").double()
    with torch.no_grad():
        router.mixing_logits.copy_(2 * torch.eye(2, dtype=torch.float64))
    streams, delta = inputs()
    torch.testing.assert_close(router(streams, delta).mean(0), 2 * streams.mean(0) + delta)
    assert router.diagnostics()["spectral_norm"].item() == pytest.approx(2.0)


@pytest.mark.parametrize("mode", MODES)
def test_gradients_reach_streams_writes_reads_and_trainable_mixer(mode):
    router = MultiStreamHyperConnection(2, mode).double()
    streams, delta = inputs()
    streams.requires_grad_()
    delta.requires_grad_()
    (router.read(streams).square().mean() + router(streams, delta).square().mean()).backward()
    for value in (streams, delta, *router.parameters()):
        if value.requires_grad:
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()
            assert value.grad.abs().max() > 1e-9
        else:
            assert value.grad is None


@pytest.mark.parametrize("dtype, device", [
    (torch.float32, "cpu"),
    pytest.param(torch.float16, "cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required for float16 softmax on PyTorch 1.13"
    )),
])
def test_diagnostics_promote_precision_and_detach(dtype, device):
    router = MultiStreamHyperConnection(2, "fixed_A0").to(device=device, dtype=dtype)
    diagnostics = router.diagnostics()
    assert all(value.dtype == torch.float32 for value in diagnostics.values())
    assert all(torch.isfinite(value) and not value.requires_grad for value in diagnostics.values())


def test_rejects_unsupported_scope_and_incompatible_shapes():
    for count in (0, 1, 3):
        with pytest.raises(ValueError, match="n_streams=2"):
            MultiStreamHyperConnection(count)
    with pytest.raises(TypeError):
        MultiStreamHyperConnection(True)
    with pytest.raises(ValueError, match="routing_mode"):
        MultiStreamHyperConnection(2, "row_stochastic")
    router = MultiStreamHyperConnection()
    streams, delta = inputs()
    with pytest.raises(ValueError, match="streams"):
        router.read(streams[:1])
    with pytest.raises(ValueError, match="delta shape"):
        router(streams, delta[:1])
