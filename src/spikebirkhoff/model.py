# Adapted from STAtten (MIT) and Spike-Driven-Transformer (Apache-2.0);
# see THIRD_PARTY_NOTICES.md; modified for static residual routing and package imports.
# Copyright (c) 2026 Zhiqi Cai

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from .layers import MS_Block_Conv, MS_SPS
from .routing import MultiStreamHyperConnection


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        chunk_size=2,
        pooling_stat="1111",
        spike_mode="lif",
        attn_mode="direct_xor",
        dvs_mode=False,
        TET=False,
        attention_mode="STAtten",
        routing="standard",
        n_streams=1,
        routing_init_scale=1e-2,
        collect_diagnostics=False,
        pretrained=False,
        pretrained_cfg=None,
        backend="cupy",
        routing_parameterization=None,
    ):
        super().__init__()
        if routing not in ("standard",) + MultiStreamHyperConnection.MODES:
            raise ValueError(f"unsupported routing mode: {routing}")
        if isinstance(n_streams, bool) or not isinstance(n_streams, int):
            raise TypeError("n_streams must be an integer")
        if routing != "standard" and n_streams != 2:
            raise ValueError("paper routing methods require n_streams=2")
        if backend not in ("torch", "cupy"):
            raise ValueError("backend must be torch or cupy")
        if routing == "standard" and n_streams != 1:
            raise ValueError("standard routing requires n_streams=1")

        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        self.attention_mode = attention_mode
        self.routing = routing
        self.n_streams = n_streams
        self.collect_diagnostics = collect_diagnostics
        self._last_routing_diagnostics = []
        self._diagnostic_state_tensors = []

        # Preserve the two frozen checkpoint formats used in the paper runs.
        if routing in ("fixed_A0", "fixed_I") and routing_parameterization is None:
            routing_parameterization = (
                "matrix" if (num_classes, depths, T) == (100, 8, 4) else "logits"
            )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
            backend=backend,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                    attention_mode=self.attention_mode,
                    chunk_size=chunk_size,
                    backend=backend,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        if routing == "standard":
            self.routing_layers = nn.ModuleList()
            self.register_buffer("input_expansion_weights", None)
        else:
            self.routing_layers = nn.ModuleList(
                [
                    MultiStreamHyperConnection(
                        n_streams=n_streams,
                        routing_mode=routing,
                        init_scale=routing_init_scale,
                        parameterization=routing_parameterization,
                    )
                    for _ in range(depths)
                ]
            )
            # A fixed, mean-preserving asymmetric expansion prevents the first
            # router from seeing identical streams. Uniform initial read
            # weights then give z_0 == x exactly, avoiding a learnable input
            # rescaling confound relative to the standard baseline.
            ramp = torch.linspace(-1.0, 1.0, n_streams)
            expansion = n_streams * torch.softmax(routing_init_scale * ramp, dim=0)
            self.register_buffer("input_expansion_weights", expansion)

            # The archive freezes the last write and mixing parameters for
            # every routed method. Its final block read remains trainable.
            # Mean readout makes the last write unidentifiable; for doubly
            # stochastic methods it also cancels the last mixing operation.
            self.routing_layers[-1].write_logits.requires_grad_(False)
            self.routing_layers[-1].mixing_logits.requires_grad_(False)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend=backend
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, _, hook = patch_embed(x, hook=hook)
        collect_now = self.collect_diagnostics and hook is not None
        self._last_routing_diagnostics = []
        self._diagnostic_state_tensors = []

        if self.routing == "standard":
            for layer_index, blk in enumerate(block):
                block_input = x
                self._retain_diagnostic_state(block_input, collect_now)
                x, _, hook = blk(x, hook=hook)
                if collect_now:
                    delta = x - block_input
                    self._last_routing_diagnostics.append(
                        {
                            "layer": layer_index,
                            "block_input_rms": self._rms(block_input),
                            "block_output_rms": self._rms(x),
                            "delta_rms": self._rms(delta),
                        }
                    )
        else:
            expansion = self.input_expansion_weights.to(
                dtype=x.dtype, device=x.device
            )
            streams = expansion.view(-1, 1, 1, 1, 1, 1) * x.unsqueeze(0)

            for layer_index, (blk, router) in enumerate(
                zip(block, self.routing_layers)
            ):
                previous_streams = streams
                z = router.read(previous_streams)
                self._retain_diagnostic_state(z, collect_now)
                block_output, _, hook = blk(z, hook=hook)
                delta = block_output - z
                streams = router(previous_streams, delta)

                if collect_now:
                    mean_error = (
                        streams.mean(0) - previous_streams.mean(0) - delta
                    )
                    route_stats = {
                        name: float(value.float().item())
                        for name, value in router.diagnostics().items()
                    }
                    route_stats.update(
                        {
                            "layer": layer_index,
                            "block_input_rms": self._rms(z),
                            "block_output_rms": self._rms(block_output),
                            "delta_rms": self._rms(delta),
                            "stream_rms": [
                                self._rms(stream) for stream in streams
                            ],
                            "stream_divergence_rms": self._rms(
                                streams - streams.mean(0, keepdim=True)
                            ),
                            "mean_residual_error_rms": self._rms(mean_error),
                            "mean_residual_error_max": float(
                                mean_error.detach().float().abs().max().item()
                            ),
                            "read_weights": router.read_weights.detach()
                            .float()
                            .cpu()
                            .tolist(),
                            "write_weights": router.write_weights.detach()
                            .float()
                            .cpu()
                            .tolist(),
                            "mixing_matrix": router.mixing_matrix()
                            .detach()
                            .float()
                            .cpu()
                            .tolist(),
                        }
                    )
                    self._last_routing_diagnostics.append(route_stats)
            x = streams.mean(0)

        x = x.flatten(3).mean(3)
        return x, hook

    @staticmethod
    def _rms(value):
        return float(value.detach().float().square().mean().sqrt().item())

    def _retain_diagnostic_state(self, value, collect_now):
        if not collect_now:
            return
        retained = None
        if torch.is_grad_enabled() and value.requires_grad:
            value.retain_grad()
            retained = value
        self._diagnostic_state_tensors.append(retained)

    @staticmethod
    def _parameter_gradient_norm(module):
        gradients = [
            parameter.grad.detach().float().norm(2)
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return None
        return float(torch.stack(gradients).norm(2).item())

    @staticmethod
    def _tensor_gradient_norm(parameter):
        if parameter.grad is None:
            return None
        return float(parameter.grad.detach().float().norm(2).item())

    def routing_diagnostics(self, state_gradient_scale=1.0):
        """Return JSON-serializable summaries from the latest logged forward.

        ``GradScaler.unscale_`` only unscales optimizer parameter gradients.
        Retained intermediate-state gradients remain scaled, so callers using
        native AMP pass the scale active during backward here.
        """
        state_gradient_scale = float(state_gradient_scale)
        if state_gradient_scale <= 0:
            raise ValueError("state_gradient_scale must be positive")
        layers = [dict(values) for values in self._last_routing_diagnostics]
        for values, state in zip(layers, self._diagnostic_state_tensors):
            gradient = None if state is None else state.grad
            if gradient is not None:
                gradient = gradient.detach().float() / state_gradient_scale
            values["state_gradient_norm"] = (
                None
                if gradient is None
                else float(gradient.norm(2).item())
            )
            values["state_gradient_rms"] = (
                None if gradient is None else self._rms(gradient)
            )
        for values, router in zip(layers, self.routing_layers):
            values["routing_gradient_norm"] = self._parameter_gradient_norm(router)
            values["read_gradient_norm"] = self._tensor_gradient_norm(
                router.read_logits
            )
            values["write_gradient_norm"] = self._tensor_gradient_norm(
                router.write_logits
            )
            values["mixing_gradient_norm"] = self._tensor_gradient_norm(
                router.mixing_logits
            )
        input_weights = None
        if self.input_expansion_weights is not None:
            input_weights = self.input_expansion_weights.detach().float().cpu().tolist()
        self._diagnostic_state_tensors = []
        return {
            "mode": self.routing,
            "n_streams": self.n_streams,
            "input_expansion_weights": input_weights,
            "layers": layers,
        }

    def no_weight_decay(self):
        """Apply one fair no-decay rule to every trainable routing logit."""
        names = set()
        for layer_index, router in enumerate(self.routing_layers):
            for parameter_name, parameter in router.named_parameters():
                if parameter.requires_grad:
                    names.add(f"routing_layers.{layer_index}.{parameter_name}")
        return names

    def forward(self, x, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
        return x, hook


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


METHODS = {
    "S": "standard", "B": "birkhoff", "U": "unconstrained",
    "Row-P": "row_projected", "DS-P": "ds_projected",
    "A0": "fixed_A0", "I": "fixed_I", "J": "fixed_J",
}


def build_model(config: dict, method: str = "B", backend: str = "cupy"):
    """Build the archived STAtten model from a flat paper/training config.

    Training-only keys are ignored. ``method`` accepts a paper label or a full
    routing mode and determines stream count. Neuron state is persistent;
    call ``spikingjelly.clock_driven.functional.reset_net`` between batches.
    """
    routing = METHODS.get(method, method)
    if routing not in ("standard",) + MultiStreamHyperConnection.MODES:
        raise ValueError(f"unsupported method: {method}")
    if config.get("model", "sdt") != "sdt":
        raise ValueError("the paper backbone is sdt")
    constructor_keys = (
        "img_size_h", "img_size_w", "patch_size", "in_channels", "num_classes",
        "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "qk_scale",
        "drop_rate", "attn_drop_rate", "drop_path_rate", "depths", "sr_ratios",
        "T", "chunk_size", "pooling_stat", "spike_mode", "attn_mode",
        "dvs_mode", "TET", "attention_mode", "routing_init_scale",
        "collect_diagnostics", "pretrained", "routing_parameterization",
    )
    kwargs = {key: config[key] for key in constructor_keys if key in config}
    aliases = {
        "time_steps": "T", "dim": "embed_dims", "layer": "depths",
        "mlp_ratio": "mlp_ratios", "drop": "drop_rate", "drop_path": "drop_path_rate",
    }
    for source, target in aliases.items():
        if source in config:
            kwargs[target] = config[source]
    if config.get("img_size") is not None:
        kwargs["img_size_h"] = kwargs["img_size_w"] = config["img_size"]
    if kwargs.get("patch_size") is None:
        kwargs["patch_size"] = 2 ** str(kwargs.get("pooling_stat", "1111")).count("1")
    if "dataset" in config:
        kwargs["dvs_mode"] = config["dataset"] in ("cifar10-dvs", "ncaltech101")
    kwargs.setdefault("depths", 4)
    kwargs.setdefault("in_channels", 3)
    kwargs.setdefault("drop_path_rate", 0.2)
    kwargs.setdefault("sr_ratios", 1)
    kwargs.update(routing=routing, n_streams=1 if routing == "standard" else 2, backend=backend)
    return sdt(**kwargs)


__all__ = ["SpikeDrivenTransformer", "sdt", "build_model", "METHODS"]
