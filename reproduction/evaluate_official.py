"""Evaluate frozen validation-selected CIFAR-100 checkpoints on the official test set."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TEST_SHA256 = '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)


def idle(gpu):
    for attempt in range(40):
        state = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = [int(value.strip()) for value in state.strip().split(',')]
        processes = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if not processes and memory < 1000 and utilization <= 10:
            return
        time.sleep(3)
    raise RuntimeError('GPU busy; no other processes will be interrupted: ' + str(gpu))


def evaluate_job(output, name):
    plan = read(output / 'evaluation_plan.json')
    record = next(row for row in plan['models'] if row['name'] == name)
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Evaluation source changed')
    if digest(__file__) != plan['evaluator_sha256']:
        raise ValueError('Evaluator changed')
    if digest(record['checkpoint']) != record['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    config = Path(record['checkpoint']).parent / 'resolved.yaml'
    if digest(config) != record['resolved_sha256']:
        raise ValueError('Resolved configuration changed')
    sys.path.insert(0, str(Path(plan['training_root']) / 'source/src'))
    from torchvision.datasets import CIFAR100
    import timm.data
    from spikebirkhoff import evaluate
    dataset = CIFAR100(plan['data_root'], train=False, download=False)
    if len(dataset) != 10000 or digest(Path(plan['data_root']) / 'cifar-100-python/test') != TEST_SHA256:
        raise ValueError('Invalid official test set')

    def official_dataset(name, root, split, is_training, **kwargs):
        if name != 'torch/cifar100' or is_training or Path(root) != Path(plan['data_root']):
            raise ValueError('Unexpected dataset request')
        return dataset

    timm.data.create_dataset = official_dataset
    write_new(output / 'receipts' / (name + '.started.json'),
              {'time': time.time(), 'checkpoint_sha256': record['checkpoint_sha256']})
    result_path = output / 'results' / (name + '.json')
    evaluate.main(['--config', str(config), '--checkpoint', record['checkpoint'],
                   '--method', record['method'], '--data-dir', plan['data_root'],
                   '--output', str(result_path), '--workers', '4', '--trusted-checkpoint'])
    result = read(result_path)
    if (result['samples'] != 10000 or not result['complete_test_set']
            or result['checkpoint_sha256'] != record['checkpoint_sha256']
            or result['checkpoint_epoch'] != record['validation_selected_epoch_zero_based']
            or result['seed'] != record['seed']):
        raise ValueError('Invalid test receipt')
    result.update(evaluation_partition='official_cifar100_test_10000', training_samples=40000,
                  validation_samples=5000, historical_official_test_exposure=True,
                  selection_note='Unchanged 5000-validation-selected checkpoint; official test not used to reselect.',
                  server=plan['server'], checkpoint_absolute_path=record['checkpoint'],
                  result_absolute_path=str(result_path), official_test_sha256=TEST_SHA256)
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    write_new(output / 'receipts' / (name + '.done.json'), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--job')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.job:
        evaluate_job(output, args.job)
        return
    root = args.root.resolve()
    gpus = [int(value) for value in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Invalid GPU list')
    protocol = read(root / 'protocol.json')
    records = read(root / 'selection_frozen_before_test.json')
    expected = {(config, method, seed) for config in protocol['configs']
                for method in protocol['methods'] for seed in protocol['seeds']}
    if len(records) != len(expected) or {(row['config'], row['method'], row['seed']) for row in records} != expected:
        raise ValueError('Incomplete frozen training matrix')
    for row in records:
        if digest(row['checkpoint']) != row['checkpoint_sha256']:
            raise ValueError('Checkpoint changed since original selection')
        row['resolved_sha256'] = digest(Path(row['checkpoint']).parent / 'resolved.yaml')
    if digest(Path(protocol['data_root']) / 'cifar-100-python/test') != TEST_SHA256:
        raise ValueError('Official test hash differs')
    for gpu in gpus:
        idle(gpu)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'receipts', 'results'):
        (output / folder).mkdir()
    plan = {'server': protocol['server'], 'training_root': str(root), 'data_root': protocol['data_root'],
            'models': records, 'gpus': gpus, 'official_test_sha256': TEST_SHA256,
            'evaluator_sha256': digest(__file__), 'selection_sha256': digest(root / 'selection_frozen_before_test.json'),
            'source_sha256': {str(path): digest(path) for path in (root / 'source/src').rglob('*.py')},
            'policy': 'Evaluate all preselected models once. No retraining, reselection or automatic retry.',
            'historical_official_test_exposure': True, 'training_samples': 40000}
    write_new(output / 'evaluation_plan.json', plan)

    def lane(index):
        results = []
        gpu = gpus[index]
        for row in records[index::len(gpus)]:
            idle(gpu)
            env = dict(os.environ)
            prefix = str(Path(sys.executable).parent.parent)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='0', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                       CUDA_PATH=prefix, LD_LIBRARY_PATH=prefix + '/lib:' + env.get('LD_LIBRARY_PATH', ''),
                       PYTHONPATH=str(root / 'source/src'))
            with (output / 'logs' / (row['name'] + '.log')).open('x') as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                                '--job', row['name']], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            results.append(dict(row, official_test=read(output / 'receipts' / (row['name'] + '.done.json'))))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = [row for group in pool.map(lane, range(len(gpus))) for row in group]
    write_new(output / 'RESULTS.json', {'server': protocol['server'], 'per_run': results,
              'evaluation_partition': 'official_cifar100_test_10000', 'training_samples': 40000,
              'historical_official_test_exposure': True})
    print('OFFICIAL_TEST_COMPLETE ' + str(output / 'RESULTS.json'), flush=True)


if __name__ == '__main__':
    main()
