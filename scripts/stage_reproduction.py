"""Copy a paper protocol into a new workspace; never launch training."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys

REPO = Path(__file__).resolve().parents[1]
DATASETS = ('cifar100', 'cifar10', 'ncaltech101', 'cifar10dvs')


def stage(dataset, data, workspace):
    data = Path(data).resolve()
    workspace = Path(workspace).resolve()
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset')
    if not data.is_dir():
        raise ValueError('An existing dataset directory is required')
    if workspace.exists():
        raise ValueError('Workspace must not exist; previous runs are never overwritten')
    if data == workspace or data in workspace.parents or REPO in workspace.parents:
        raise ValueError('Workspace must be outside the dataset and repository')
    workspace.mkdir(parents=True)
    if dataset == 'cifar10dvs':
        shutil.copytree(REPO / 'reproduction/cifar10dvs', workspace / 'code')
    else:
        template = REPO / 'reproduction/protocols' / dataset
        for source in template.iterdir():
            if source.is_file():
                shutil.copy2(source, workspace / source.name)
        shutil.copytree(REPO / 'reproduction/frozen_core', workspace / 'source')
        protocol_path = workspace / 'protocol.json'
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        protocol.update(data_root=str(data), remote_root=str(workspace),
                        python=sys.executable, server=socket.gethostname())
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = stage(args.dataset, args.data, args.workspace)
    print('Staged:', workspace)
    print('No training or evaluation started. See docs/reproduction.md before launching.')


if __name__ == '__main__':
    main()
