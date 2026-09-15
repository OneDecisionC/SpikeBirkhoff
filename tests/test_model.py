# Copyright (c) 2026 Zhiqi Cai
# SPDX-License-Identifier: MIT
"""CPU checks against the actual spiking backbone and optimizer grouping."""

import pytest
import torch
from spikingjelly.clock_driven import functional
from timm.optim import create_optimizer_v2

from spikebirkhoff.model import SpikeDrivenTransformer, build_model
from spikebirkhoff.routing import MultiStreamHyperConnection


def small_model(routing="standard", depths=3, **kwargs):
    return SpikeDrivenTransformer(
        img_size_h=8, img_size_w=8, patch_size=4, in_channels=3,
        num_classes=5, embed_dims=16, num_heads=2, mlp_ratios=2,
        depths=depths, T=2, chunk_size=1, pooling_stat="0011",
        spike_mode="lif", attention_mode="STAtten", routing=routing,
        n_streams=1 if routing == "standard" else 2,
        routing_init_scale=0.01, backend="torch", **kwargs,
    )


@pytest.mark.parametrize("mode", MultiStreamHyperConnection.MODES)
def test_model_initialization_keeps_backbone_weights_and_rng_paired(mode):
    torch.manual_seed(31)
    standard = small_model()
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(31)
    routed = small_model(mode)
    assert torch.equal(reference_rng, torch.get_rng_state())
    state = routed.state_dict()
    for name, expected in standard.state_dict().items():
        torch.testing.assert_close(state[name], expected, rtol=0, atol=0)
    assert all(module.backend == "torch" for module in routed.modules() if hasattr(module, "backend"))


@pytest.mark.parametrize("mode", MultiStreamHyperConnection.MODES)
def test_uniform_reads_match_standard_features_with_identical_backbone(mode):
    torch.manual_seed(37)
    baseline = small_model().train()
    routed = small_model(mode).train()
    loaded = routed.load_state_dict(baseline.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert all(name == "input_expansion_weights" or name.startswith("routing_layers.") for name in loaded.missing_keys)
    inputs = torch.randn(2, 2, 3, 8, 8)
    functional.reset_net(baseline)
    functional.reset_net(routed)
    with torch.no_grad():
        expected, _ = baseline.forward_features(inputs)
        actual, _ = routed.forward_features(inputs)
    # Features avoid a vacuous equality of silent untrained classifier spikes.
    assert expected.square().mean() > 1e-5
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-6)


@pytest.mark.parametrize("mode", MultiStreamHyperConnection.MODES)
def test_freeze_policy_and_actual_timm_weight_decay_groups(mode):
    model = small_model(mode)
    for index, router in enumerate(model.routing_layers):
        assert router.read_logits.requires_grad
        assert router.write_logits.requires_grad == (index < model.depths - 1)
        expected_mixing = index < model.depths - 1 and mode not in ("fixed_A0", "fixed_I", "fixed_J")
        assert router.mixing_logits.requires_grad == expected_mixing
    expected_names = {
        name for name, parameter in model.named_parameters()
        if name.startswith("routing_layers.") and parameter.requires_grad
    }
    assert model.no_weight_decay() == expected_names
    optimizer = create_optimizer_v2(model, opt="adamw", lr=3e-4, weight_decay=0.06)
    routing_ids = {id(p) for name, p in model.named_parameters() if name in expected_names}
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in routing_ids:
                assert group["weight_decay"] == 0
                seen.add(id(parameter))
    assert seen == routing_ids


def test_learned_final_read_and_earlier_mixing_receive_task_gradients():
    torch.manual_seed(43)
    model = small_model("birkhoff", collect_diagnostics=True).train()
    with torch.no_grad():
        for router in model.routing_layers:
            router.read_logits.copy_(torch.tensor([-0.2, 0.2]))
    functional.reset_net(model)
    features, _ = model.forward_features(torch.randn(2, 2, 3, 8, 8), hook={})
    features.square().mean().backward()
    assert model.routing_layers[-1].read_logits.grad.abs().max() > 1e-9
    assert model.routing_layers[0].mixing_logits.grad.abs().max() > 1e-9
    assert model.routing_layers[0].write_logits.grad.abs().max() > 1e-9
    assert model.routing_layers[-1].mixing_logits.grad is None
    assert model.routing_layers[-1].write_logits.grad is None
    diagnostic = model.routing_diagnostics()
    assert len(diagnostic["layers"]) == 3
    assert diagnostic["layers"][0]["stream_divergence_rms"] > 0
    assert diagnostic["layers"][0]["state_gradient_norm"] > 0


def test_cpu_forward_layouts_and_temporal_readout():
    torch.manual_seed(47)
    model = small_model("birkhoff", TET=True).train()
    images = torch.randn(2, 3, 8, 8)
    functional.reset_net(model)
    with torch.no_grad():
        static_output, _ = model(images)
    functional.reset_net(model)
    with torch.no_grad():
        event_output, _ = model(images.unsqueeze(1).repeat(1, 2, 1, 1, 1))
    assert event_output.shape == (2, 2, 5)
    torch.testing.assert_close(event_output, static_output)
    model.TET = False
    functional.reset_net(model)
    with torch.no_grad():
        averaged_output, _ = model(images)
    torch.testing.assert_close(averaged_output, static_output.mean(0))


def test_archived_drop_path_is_inactive_even_during_training():
    torch.manual_seed(53)
    baseline = small_model(drop_path_rate=0.0).train()
    changed = small_model(drop_path_rate=0.9).train()
    changed.load_state_dict(baseline.state_dict())
    inputs = torch.randn(2, 2, 3, 8, 8)
    functional.reset_net(baseline)
    functional.reset_net(changed)
    with torch.no_grad():
        expected, _ = baseline.forward_features(inputs)
        actual, _ = changed.forward_features(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("method, mode", [
    ("S", "standard"), ("B", "birkhoff"), ("U", "unconstrained"),
    ("Row-P", "row_projected"), ("DS-P", "ds_projected"),
    ("A0", "fixed_A0"), ("I", "fixed_I"), ("J", "fixed_J"),
])
def test_build_model_accepts_archived_config_keys(method, mode):
    config = {
        "model": "sdt", "img_size": 8, "dim": 16, "num_heads": 2,
        "layer": 2, "time_steps": 2, "mlp_ratio": 2, "chunk_size": 1,
        "pooling_stat": "0011", "num_classes": 5, "in_channels": 3,
        "batch_size": 8,
    }
    model = build_model(config, method=method, backend="torch")
    assert model.routing == mode
    assert model.n_streams == (1 if method == "S" else 2)
    assert model.depths == 2
    assert model.T == 2
    assert model.patch_embed.patch_size == (4, 4)


@pytest.mark.parametrize("method", ("A0", "I"))
def test_fixed_parameterization_matches_paper_configuration(method):
    config = {
        "img_size": 8, "dim": 16, "num_heads": 2, "layer": 8,
        "time_steps": 4, "num_classes": 100,
    }
    matrix_model = build_model(config, method=method, backend="torch")
    assert matrix_model.routing_layers[0].mixing_logits.shape == (2, 2)
    config["time_steps"] = 2
    logits_model = build_model(config, method=method, backend="torch")
    assert logits_model.routing_layers[0].mixing_logits.shape == (2,)


def test_build_model_infers_event_attention_from_dataset():
    model = build_model({
        "dataset": "ncaltech101", "img_size": 8, "dim": 16,
        "num_heads": 2, "layer": 2, "time_steps": 2,
        "pooling_stat": "0011", "in_channels": 2,
    }, backend="torch")
    assert model.dvs
    assert all(block.attn.dvs for block in model.block)
    assert model.patch_embed.patch_size == (4, 4)


def test_training_call_and_evaluation_builder_match_all_paper_configs(monkeypatch):
    """Read the actual training call without importing its GPU/data runtime."""
    import ast
    import inspect
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    import spikebirkhoff.model as model_module

    package = Path(model_module.__file__).parent
    tree = ast.parse((package / "training_engine.py").read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            options = {item.arg: item.value for item in node.keywords}
            if "default" in options and node.args:
                flag = ast.literal_eval(node.args[0])
                if flag.startswith("--"):
                    name = ast.literal_eval(options["dest"]) if "dest" in options else flag[2:].replace("-", "_")
                    try:
                        defaults[name] = ast.literal_eval(options["default"])
                    except (ValueError, TypeError):
                        pass  # Unrelated augmentation defaults contain arithmetic.
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "create_model")
    constructor_defaults = {
        name: parameter.default for name, parameter in inspect.signature(SpikeDrivenTransformer).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    monkeypatch.setattr(model_module, "sdt", lambda **kwargs: kwargs)
    configs = sorted((package.parents[1] / "configs").glob("*.yaml"))
    assert len(configs) == 7
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        arguments = dict(defaults, **config)
        arguments["backend"] = "torch"
        arguments["dvs_mode"] = arguments["dataset"] in ("cifar10-dvs", "ncaltech101")
        if arguments.get("patch_size") is None:
            arguments["patch_size"] = 2 ** str(arguments["pooling_stat"]).count("1")
        training_kwargs = eval(compile(ast.Expression(call), str(path), "eval"), {
            "args": SimpleNamespace(**arguments),
            "create_model": lambda name, **kwargs: kwargs,
        })
        training_kwargs.pop("drop_block_rate")  # timm filters this unused None option.
        evaluation_kwargs = model_module.build_model(config, method="B", backend="torch")
        assert dict(constructor_defaults, **training_kwargs) == dict(constructor_defaults, **evaluation_kwargs), path.name


def test_rejects_multiple_standard_streams_and_unknown_backend():
    with pytest.raises(ValueError, match="standard routing requires"):
        SpikeDrivenTransformer(routing="standard", n_streams=2, depths=2)
    with pytest.raises(ValueError, match="backend"):
        SpikeDrivenTransformer(backend="unknown", depths=2)
