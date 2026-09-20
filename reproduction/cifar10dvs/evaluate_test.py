"""Wait for all frozen training runs; evaluate each selected checkpoint once."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
RUNS = None
sys.path[:0] = [str(CODE), str(CODE / 'vendor/src')]
from common import assert_idle, environment, read_json, verify_freeze, verify_selections, write_new
from rental_data import RecordingDataset, sha256


def evaluate(name):
    import numpy as np
    import torch
    from spikingjelly.clock_driven import functional
    from spikebirkhoff.config import load_config
    from spikebirkhoff.model import build_model
    from timm.models.helpers import clean_state_dict
    selected = read_json(RUNS / 'selection_frozen_before_test.json')
    record = next(row for row in selected if row['name'] == name)
    gate = read_json(RUNS / 'test_gate.json')
    if gate['selection_sha256'] != sha256(RUNS / 'selection_frozen_before_test.json'):
        raise ValueError('Frozen selections changed')
    if gate['evaluator_sha256'] != sha256(__file__):
        raise ValueError('Evaluator changed')
    freeze = verify_freeze(RUNS)
    checkpoint = Path(record['checkpoint'])
    resolved = checkpoint.parent / 'resolved.yaml'
    if sha256(checkpoint) != record['checkpoint_sha256'] or sha256(resolved) != record['resolved_sha256']:
        raise ValueError('Selected checkpoint or configuration changed')
    cfg = load_config(resolved, record['method'])
    if cfg['seed'] != record['seed'] or not cfg['amp'] or not cfg['TET']:
        raise ValueError('Unexpected evaluation protocol')
    write_new(RUNS / 'test_results' / (name + '.started.json'),
              {'checkpoint': str(checkpoint), 'time': time.time(), 'evaluator_sha256': sha256(__file__)})
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(checkpoint, map_location='cpu')
    if payload['epoch'] != record['validation_selected_epoch_zero_based']:
        raise ValueError('Checkpoint epoch does not match frozen validation selection')
    model = build_model(cfg, method=record['method'], backend='cupy')
    model.load_state_dict(clean_state_dict(payload['state_dict']), strict=True)
    model.cuda().eval()
    dataset = RecordingDataset(freeze['data_absolute_path'], RUNS / 'split.json', 'internal_test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['val_batch_size'], shuffle=False,
                                        num_workers=4)
    total = correct = 0
    functional.reset_net(model)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.float().half().cuda(), targets.cuda()
            try:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)[0]
                logits = logits.mean(0)
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite logits')
                correct += logits.argmax(1).eq(targets).sum().item()
                total += targets.numel()
            finally:
                functional.reset_net(model)
    if total != 1000 or sha256(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Incomplete test or changed checkpoint')
    write_new(RUNS / 'test_results' / (name + '.json'), dict(record, test_top1=100 * correct / total,
              samples=total, complete_test_set=True, evaluation_partition='internal_test',
              split_sha256=freeze['split_sha256'], checkpoint_epoch=payload['epoch'],
              evaluator_sha256=sha256(__file__), server=os.environ.get('HOSTNAME', 'unknown')))


def wait_idle(gpu):
    deadline = time.monotonic() + 120
    while True:
        try:
            assert_idle(gpu)
            return
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(3)


def main():
    global RUNS
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--trusted-checkpoints', action='store_true')
    args = parser.parse_args()
    if not args.trusted_checkpoints:
        parser.error('Only load your own trusted checkpoints; pass --trusted-checkpoints.')
    RUNS = args.runs.resolve()
    gpus = [int(value) for value in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        parser.error('GPU indices must be unique nonnegative integers.')
    if args.preflight:
        import torch
        from spikebirkhoff.config import load_config
        from spikebirkhoff.model import build_model
        from timm.models.helpers import clean_state_dict
        verify_freeze(RUNS)
        record = read_json(next((RUNS / 'receipts').glob('*.done.json')))
        checkpoint = torch.load(record['checkpoint'], map_location='cpu')
        assert checkpoint['epoch'] == record['validation_selected_epoch_zero_based']
        cfg = load_config(Path(record['checkpoint']).parent / 'resolved.yaml', record['method'])
        model = build_model(cfg, method=record['method'], backend='torch')
        model.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=True)
        print('PREFLIGHT_OK: checkpoint reload on CPU; no test data accessed.')
        return
    if args.job:
        evaluate(args.job)
        return
    with (RUNS / 'final_test.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (RUNS / 'TRAINING_COMPLETE.json').exists():
            raise RuntimeError('All 20 training runs must finish before test evaluation.')
        verify_freeze(RUNS)
        protocol = read_json(CODE / 'protocol.json')
        records = verify_selections(RUNS, protocol)
        (RUNS / 'test_results').mkdir(exist_ok=True)
        write_new(RUNS / 'test_gate.json', {'selection_sha256': sha256(RUNS / 'selection_frozen_before_test.json'),
                  'evaluator_sha256': sha256(__file__), 'time': time.time(), 'count': len(records)})

        def lane(index):
            gpu = gpus[index]
            results = []
            for record in records[index::len(gpus)]:
                name = record['name']
                wait_idle(gpu)
                with (RUNS / 'test_results' / (name + '.log')).open('x') as stream:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--job', name,
                                    '--runs', str(RUNS), '--trusted-checkpoints'],
                                   env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT, check=True)
                results.append(read_json(RUNS / 'test_results' / (name + '.json')))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            results = [row for group in pool.map(lane, range(len(gpus))) for row in group]
        if len(results) != 20:
            raise ValueError('Incomplete final results')
        summary = {'per_run': results, 'aggregates': [], 'paired_differences': [],
                   'evaluation_partition': 'internal_test_1000', 'historical_dataset_exposure': True}
        for method in protocol['methods']:
            values = [row['test_top1'] for row in results if row['method'] == method]
            summary['aggregates'].append({'method': method, 'n': len(values),
                                         'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)})
        indexed = {(row['method'], row['seed']): row['test_top1'] for row in results}
        for control in ('A0', 'S', 'I'):
            values = [indexed['B', seed] - indexed[control, seed] for seed in protocol['seeds']]
            summary['paired_differences'].append({'contrast': 'B-' + control, 'values_pp': values,
                    'mean_pp': statistics.mean(values), 'sample_sd_pp': statistics.stdev(values)})
        write_new(RUNS / 'RESULTS.json', summary)
        print('FINAL_TEST_COMPLETE ' + str(RUNS / 'RESULTS.json'), flush=True)


if __name__ == '__main__':
    main()
