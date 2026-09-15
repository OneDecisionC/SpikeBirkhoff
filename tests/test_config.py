from pathlib import Path
import pytest
from spikebirkhoff.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_and_method_override():
    profiles = list((ROOT / "configs").glob("*.yaml"))
    assert len(profiles) == 7
    for path in profiles:
        config = load_config(path, "S")
        assert config["n_streams"] == 1
        assert config["seed"] == 42
        assert config["epochs"] + config["cooldown_epochs"] == 210
        assert load_config(path, "I")["routing"] == "fixed_I"


def test_reject_unknown_method_and_invalid_chunk():
    path = ROOT / "configs/cifar10_d6_t4.yaml"
    with pytest.raises(ValueError):
        load_config(path, "unknown")
    with pytest.raises(ValueError):
        load_config(path, "B", ["chunk_size=3"])
