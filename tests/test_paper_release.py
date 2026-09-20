import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_statistics():
    rows = load_script('summarize_paper').summarize()
    paired = {row['config']: row for row in rows if row['partition'] == 'O' and row['method'] == 'B-A0'}
    assert paired['cifar100_d8_t4']['mean'] == pytest.approx(0.014)
    assert paired['cifar100_d16_t4']['mean'] == pytest.approx(-0.058)


@pytest.mark.parametrize('dataset', ('cifar100', 'cifar10', 'ncaltech101', 'cifar10dvs'))
def test_stage_never_launches_or_overwrites(tmp_path, dataset):
    data = tmp_path / 'data'
    data.mkdir()
    workspace = tmp_path / 'run'
    staging = load_script('stage_reproduction')
    staging.stage(dataset, data, workspace)
    assert not (workspace / 'launch_receipt.json').exists()
    protocol = workspace / ('code/protocol.json' if dataset == 'cifar10dvs' else 'protocol.json')
    assert json.loads(protocol.read_text())['seeds'] == [42, 43, 44, 45, 46]
    with pytest.raises(ValueError, match='must not exist'):
        staging.stage(dataset, data, workspace)


def test_reject_dataset_subdirectory(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        load_script('stage_reproduction').stage('cifar100', tmp_path, tmp_path / 'run')
