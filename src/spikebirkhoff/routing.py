# Copyright (c) 2026 Zhiqi Cai
# SPDX-License-Identifier: MIT

"""Multi-stream residual routing for SpikeBirkhoff experiments.

The router keeps stream routing separate from the wrapped network block.  A
caller first obtains the block input with :meth:`read`, computes a residual
update ``delta``, and then calls the module to mix the streams and write the
update back::

    z = router.read(X)
    delta = block(z) - z
    X = router(X, delta)

``X`` has shape ``[S, T, B, C, H, W]`` and ``delta`` has shape
``[T, B, C, H, W]``.
"""

from typing import Dict

import torch
from torch import Tensor, nn


class MultiStreamHyperConnection(nn.Module):
    """Static residual routing for the paper's two-stream experiments.

    ``birkhoff`` learns two logits over identity and swap. ``unconstrained``,
    ``row_projected``, and ``ds_projected`` store a raw 2-by-2 matrix; the latter
    two apply the exact Euclidean projections at each forward. The fixed
    controls freeze only the mixing parameters, retaining learned reads and
    writes. Model-level code additionally freezes the final write and mixer.

    Initialization uses no random draws. Parameter names and raw shapes match
    the archived runtimes: B/J use two logits; U/Row-P/DS-P use matrices.
    Fixed A0/I had both archived forms, selected by ``parameterization``.
    """

    MODES = (
        "birkhoff", "unconstrained", "row_projected", "ds_projected",
        "fixed_A0", "fixed_I", "fixed_J",
    )

    def __init__(
        self,
        n_streams: int = 2,
        routing_mode: str = "birkhoff",
        init_scale: float = 0.05,
        *,
        mode: str = None,
        parameterization: str = None,
    ) -> None:
        super().__init__()
        if isinstance(n_streams, bool) or not isinstance(n_streams, int):
            raise TypeError("n_streams must be an integer")
        if n_streams != 2:
            raise ValueError("paper routing methods require n_streams=2")
        if mode is not None:
            if routing_mode != "birkhoff" and routing_mode != mode:
                raise ValueError("routing_mode and mode specify different values")
            routing_mode = mode
        if routing_mode not in self.MODES:
            raise ValueError("routing_mode must be one of " + ", ".join(self.MODES))
        if init_scale < 0:
            raise ValueError("init_scale must be non-negative")
        if parameterization not in (None, "logits", "matrix"):
            raise ValueError("parameterization must be logits or matrix")
        if parameterization is not None and routing_mode not in ("fixed_A0", "fixed_I"):
            raise ValueError("parameterization only applies to fixed_A0 and fixed_I")
        if parameterization is None:
            parameterization = "matrix" if routing_mode == "fixed_A0" else "logits"

        self.n_streams = n_streams
        self.routing_mode = routing_mode
        ramp = torch.linspace(-1.0, 1.0, n_streams)
        self.read_logits = nn.Parameter(torch.zeros_like(ramp))
        self.write_logits = nn.Parameter(-init_scale * ramp)
        initial_logits = init_scale * ramp
        initial_logits = initial_logits.clone()
        initial_logits[0] += 2.0

        if routing_mode == "unconstrained":
            # Preserve the archived operation order for effective A0 pairing.
            coefficients = torch.softmax(initial_logits, dim=0)
            initial_permutations = torch.stack(
                [torch.roll(torch.eye(n_streams), shifts=k, dims=1)
                 for k in range(n_streams)], dim=0,
            )
            initial_mixing = torch.einsum(
                "k,kij->ij", coefficients, initial_permutations
            )
            self.mixing_logits = nn.Parameter(initial_mixing)
        elif routing_mode in ("row_projected", "ds_projected") or (
            routing_mode in ("fixed_A0", "fixed_I") and parameterization == "matrix"
        ):
            coefficients = torch.softmax(initial_logits, dim=0)
            initial_mixing = (
                coefficients[0] * torch.eye(2)
                + coefficients[1] * torch.flip(torch.eye(2), dims=[1])
            )
            self.mixing_logits = nn.Parameter(
                initial_mixing, requires_grad=routing_mode not in ("fixed_A0", "fixed_I")
            )
        else:
            self.mixing_logits = nn.Parameter(
                initial_logits, requires_grad=routing_mode == "birkhoff"
            )

        permutations = torch.stack(
            [torch.roll(torch.eye(n_streams), shifts=k, dims=1)
             for k in range(n_streams)], dim=0,
        )
        self.register_buffer("cyclic_permutations", permutations, persistent=False)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        # A0/I were archived as both frozen logits and frozen matrices. Keep
        # checkpoint values and raw shapes, without converting or reinitializing.
        saved = state_dict.get(prefix + "mixing_logits")
        if self.routing_mode in ("fixed_A0", "fixed_I") and saved is not None:
            if tuple(saved.shape) in ((2,), (2, 2)) and saved.shape != self.mixing_logits.shape:
                self.mixing_logits = nn.Parameter(
                    self.mixing_logits.new_empty(saved.shape), requires_grad=False
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    @property
    def mode(self) -> str:
        """Alias retained for concise experiment configuration."""
        return self.routing_mode

    @property
    def read_weights(self) -> Tensor:
        """Convex read weights, with shape ``[S]`` and sum one."""
        return torch.softmax(self.read_logits, dim=0)

    @property
    def write_weights(self) -> Tensor:
        """Write weights with shape ``[S]`` and sum ``S``."""
        return self.n_streams * torch.softmax(self.write_logits, dim=0)

    def mixing_matrix(self) -> Tensor:
        """Return the effective stream mixing matrix ``A``."""
        w = self.mixing_logits
        if self.routing_mode == "unconstrained" or (
            self.routing_mode == "fixed_A0" and w.ndim == 2
        ):
            return w
        if self.routing_mode == "fixed_I":
            return torch.eye(2, device=w.device, dtype=w.dtype)
        if self.routing_mode == "fixed_J":
            return w.new_full((2, 2), 0.5)
        if self.routing_mode == "row_projected":
            a = ((w[:, 0] - w[:, 1] + 1.0) * 0.5).clamp(0.0, 1.0)
            return torch.stack((a, 1.0 - a), dim=1)
        if self.routing_mode == "ds_projected":
            a = (
                (w[0, 0] + w[1, 1] - w[0, 1] - w[1, 0] + 2.0) * 0.25
            ).clamp(0.0, 1.0)
            return torch.stack((torch.stack((a, 1.0-a)), torch.stack((1.0-a, a))))

        coefficients = torch.softmax(w, dim=0)
        permutations = self.cyclic_permutations.to(
            dtype=coefficients.dtype, device=coefficients.device
        )
        return torch.einsum("k,kij->ij", coefficients, permutations)

    def _validate_X(self, X: Tensor) -> None:
        if X.ndim != 6:
            raise ValueError("X must have shape [S, T, B, C, H, W]")
        if X.shape[0] != self.n_streams:
            raise ValueError(
                f"X has {X.shape[0]} streams, expected {self.n_streams}"
            )

    def read(self, X: Tensor) -> Tensor:
        """Read a convex combination of streams for the wrapped block."""
        self._validate_X(X)
        weights = self.read_weights.to(dtype=X.dtype, device=X.device)
        return torch.einsum("s,stbchw->tbchw", weights, X)

    def mix(self, X: Tensor) -> Tensor:
        """Apply only the stream-to-stream mixing operation."""
        self._validate_X(X)
        matrix = self.mixing_matrix().to(dtype=X.dtype, device=X.device)
        return torch.einsum("ij,jtbchw->itbchw", matrix, X)

    def write(self, delta: Tensor) -> Tensor:
        """Expand a residual update into its per-stream write contributions."""
        if delta.ndim != 5:
            raise ValueError("delta must have shape [T, B, C, H, W]")
        weights = self.write_weights.to(dtype=delta.dtype, device=delta.device)
        return torch.einsum("s,tbchw->stbchw", weights, delta)

    def forward(self, X: Tensor, delta: Tensor) -> Tensor:
        """Compute ``A @ X + q * delta``."""
        self._validate_X(X)
        if delta.ndim != 5:
            raise ValueError("delta must have shape [T, B, C, H, W]")
        if tuple(delta.shape) != tuple(X.shape[1:]):
            raise ValueError(
                "delta shape must equal X.shape[1:] (the non-stream dimensions)"
            )
        return self.mix(X) + self.write(delta)

    def diagnostics(self, detach: bool = True) -> Dict[str, Tensor]:
        """Return inexpensive constraint and routing summaries.

        The returned values remain tensors so callers can log them without
        losing device information.  By default they are detached from autograd.
        """
        # Diagnostics run inside the training autocast context. In Birkhoff
        # mode the einsum that forms the matrix is therefore commonly FP16,
        # while spectral matrix norms do not support low-precision CUDA/CPU
        # inputs on the PyTorch versions used by this project. Keep routing
        # itself in autocast, but perform all logging reductions in FP32.
        matrix = self.mixing_matrix().float()
        read = self.read_weights.float()
        write = self.write_weights.float()
        values = {
            "row_sum_error": (matrix.sum(dim=1) - 1).abs().max(),
            "column_sum_error": (matrix.sum(dim=0) - 1).abs().max(),
            "minimum_entry": matrix.min(),
            "read_entropy": -(read * read.clamp_min(1e-12).log()).sum(),
            "write_sum": write.sum(),
            "spectral_norm": torch.linalg.matrix_norm(matrix, ord=2),
        }
        if detach:
            return {name: value.detach() for name, value in values.items()}
        return values


__all__ = ["MultiStreamHyperConnection"]
