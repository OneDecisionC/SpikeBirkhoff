"""Validation-only paired interventions on frozen, selected B checkpoints."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def idle(gpu):
    for attempt in range(40):
        processes = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        usage = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(int, usage.strip().split(','))
        if not processes and memory < 1000 and utilization <= 10:
            return
        time.sleep(3)
    raise RuntimeError('GPU occupied; not interrupting it: ' + str(gpu))


def worker(args, plan):
    import random
    import numpy as np
    import torch
    source = Path(plan['source'])
    sys.path[:0] = [str(source), str(Path(plan['adapter']))]
    from spikebirkhoff.config import load_config
    from spikebirkhoff.model import build_model
    from timm.models.helpers import clean_state_dict
    from spikingjelly.clock_driven import functional, neuron
    record = next(row for row in plan['records'] if row['name'] == args.job)
    checkpoint = Path(record['local_checkpoint'])
    if digest(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint hash changed')
    cfg = load_config(checkpoint.parent / 'resolved.yaml', 'B')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(checkpoint, map_location='cpu')
    if payload['epoch'] != record['validation_selected_epoch_zero_based']:
        raise ValueError('Selected epoch changed')
    model = build_model(cfg, method='B', backend='cupy')
    initial = [router.mixing_logits.detach().clone() for router in model.routing_layers]
    initial_matrices = [router.mixing_matrix().detach().clone() for router in model.routing_layers]
    model.load_state_dict(clean_state_dict(payload['state_dict']), strict=True)
    del payload
    model.cuda().eval()
    original = [{key: value.detach().clone() for key, value in router.state_dict().items()}
                for router in model.routing_layers]
    modes = [router.routing_mode for router in model.routing_layers]
    parameters = []
    for index, router in enumerate(model.routing_layers):
        matrix = router.mixing_matrix().detach().float().cpu()
        parameters.append({'layer': index, 'read': router.read_weights.detach().cpu().tolist(),
            'write': router.write_weights.detach().cpu().tolist(), 'matrix': matrix.tolist(),
            'alpha_change': float(matrix[0, 0] - initial_matrices[index][0, 0]),
            'matrix_delta_fro': float((matrix - initial_matrices[index]).norm()),
            'requires_grad': {name: value.requires_grad for name, value in router.named_parameters()}})
    bn = {name: value.detach().cpu().clone() for name, value in model.named_buffers()
          if 'running_' in name or 'num_batches_tracked' in name}
    if plan['dataset'] == 'cifar100':
        from holdout import load_partition
        from timm.data import create_loader, resolve_data_config
        dataset = load_partition(plan['root'], 'validation')
        data_cfg = resolve_data_config(cfg, model=model)
        loader = create_loader(dataset, input_size=data_cfg['input_size'], batch_size=cfg.get('val_batch_size', 64),
            is_training=False, use_prefetcher=False, interpolation=data_cfg['interpolation'],
            mean=data_cfg['mean'], std=data_cfg['std'], crop_pct=data_cfg['crop_pct'], num_workers=4)
        per_class = 5
    else:
        from rental_data import RecordingDataset
        dataset = RecordingDataset(plan['data'], Path(plan['root']) / 'split.json', 'validation')
        for row in dataset.records:
            if digest(dataset.root / row['relative']) != row['sha256']:
                raise ValueError('DVS validation data changed')
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.get('val_batch_size', cfg['batch_size']),
                                            shuffle=False, num_workers=4)
        per_class = 20
    counts = {}
    probe_indices = []
    for index, target in enumerate(dataset.targets):
        if counts.get(target, 0) < per_class:
            probe_indices.append(index)
            counts[target] = counts.get(target, 0) + 1
    probe_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, probe_indices),
                                               batch_size=8, shuffle=False, num_workers=2)
    variants = ('B', 'restore_A0', 'identity', 'uniform_read')

    def intervene(variant):
        for index, router in enumerate(model.routing_layers):
            router.load_state_dict(original[index], strict=True)
            router.routing_mode = modes[index]
            if variant == 'restore_A0':
                router.mixing_logits.copy_(initial[index].to(router.mixing_logits))
            elif variant == 'identity':
                router.routing_mode = 'fixed_I'
            elif variant == 'uniform_read':
                router.read_logits.zero_()

    def forward(inputs):
        functional.reset_net(model)
        try:
            inputs = inputs.float().cuda()
            if cfg.get('amp', False):
                inputs = inputs.half()
            with torch.cuda.amp.autocast(enabled=cfg.get('amp', False)):
                output = model(inputs)[0]
            if cfg.get('TET', False):
                output = output.mean(0)
            if not torch.isfinite(output).all():
                raise ValueError('Nonfinite logits')
            return output.float().cpu()
        finally:
            functional.reset_net(model)

    started = time.time()
    results = {}
    logits = {}
    with torch.no_grad():
        for variant in variants:
            intervene(variant)
            outputs, labels = [], []
            for batch, (inputs, targets) in enumerate(loader):
                outputs.append(forward(inputs))
                labels.append(targets)
                if args.smoke:
                    break
            logits[variant] = torch.cat(outputs)
            targets = torch.cat(labels)
            output = logits[variant]
            baseline = logits['B']
            results[variant] = {'samples': len(targets),
                'accuracy': float(output.argmax(1).eq(targets).float().mean() * 100),
                'prediction_disagreement': float(output.argmax(1).ne(baseline.argmax(1)).float().mean()),
                'logit_relative_fro': float((output - baseline).norm() / baseline.norm().clamp_min(1e-12)),
                'probability_mean_abs_change': float((output.softmax(1) - baseline.softmax(1)).abs().mean())}
            print(json.dumps({'variant': variant, **results[variant]}), flush=True)
        capture = {'active': False}
        saved_reads = []
        for index, router in enumerate(model.routing_layers):
            original_read = router.read
            saved_reads.append(original_read)

            def tracked_read(streams, index=index, original_read=original_read):
                output = original_read(streams)
                if capture['active']:
                    average = streams.float().mean(0)
                    contrast = streams[0].float() - streams[1].float()
                    denominator = average.norm().clamp_min(1e-12)
                    capture['z'][str(index)] = output.detach().float().cpu()
                    capture['cr'][str(index)] = [float(contrast.norm() / denominator),
                                                 float((output.float() - average).norm() / denominator)]
                return output

            router.read = tracked_read
        handles = []
        spike_sites = []
        for name, module in model.named_modules():
            if isinstance(module, neuron.BaseNode) and name.startswith('block'):
                spike_sites.append(name)

                def track_spikes(module, inputs, output, name=name):
                    if capture['active']:
                        capture['spikes'][name] = output.detach().bool().cpu()

                handles.append(module.register_forward_hook(track_spikes))
        if not spike_sites:
            raise ValueError('No block spike probe sites matched')
        probes = {variant: {} for variant in variants}
        cr = {}
        repeat_max = 0.0
        for batch, (inputs, targets) in enumerate(probe_loader):
            reference = None
            for variant in variants:
                intervene(variant)
                capture.update(active=True, z={}, spikes={}, cr={})
                current_logits = forward(inputs)
                capture['active'] = False
                current = {'z': capture['z'], 'spikes': capture['spikes']}
                if reference is None:
                    reference = current
                    repeated = forward(inputs)
                    repeat_max = max(repeat_max, float((current_logits - repeated).abs().max()))
                    if not torch.equal(current_logits, repeated):
                        raise ValueError('Repeated baseline forward differs')
                    for name, values in capture['cr'].items():
                        totals = cr.setdefault(name, [0.0, 0.0, 0])
                        totals[0] += values[0] * len(targets)
                        totals[1] += values[1] * len(targets)
                        totals[2] += len(targets)
                for name, value in current['z'].items():
                    totals = probes[variant].setdefault('z.' + name, [0.0, 0.0])
                    totals[0] += float((value - reference['z'][name]).square().sum())
                    totals[1] += float(reference['z'][name].square().sum())
                for name, value in current['spikes'].items():
                    totals = probes[variant].setdefault('spike.' + name, [0, 0])
                    totals[0] += int(value.ne(reference['spikes'][name]).sum())
                    totals[1] += value.numel()
            if args.smoke:
                break
        for handle in handles:
            handle.remove()
        for name, value in model.named_buffers():
            if name in bn and not torch.equal(bn[name], value.cpu()):
                raise ValueError('BN buffers changed')
    result = {'server': plan['server'], 'checkpoint_absolute_path': str(checkpoint),
        'checkpoint_sha256': record['checkpoint_sha256'], 'original_checkpoint_path': record['checkpoint'],
        'result_absolute_path': str(Path(args.output) / (args.job + '.json')),
        'plan_sha256': digest(Path(args.output) / 'plan.json'), 'evaluator_sha256': digest(__file__),
        'split_sha256': digest(Path(plan['root']) / 'split.json'),
        'resolved_sha256': digest(checkpoint.parent / 'resolved.yaml'),
        'seed': cfg['seed'], 'epoch': record['validation_selected_epoch_zero_based'],
        'partition': 'validation', 'smoke_only': args.smoke, 'parameters': parameters,
        'metrics': results, 'probe_indices': probe_indices, 'probe_batch_size': 8,
        'probe_sums': probes, 'baseline_C_R_sample_weighted_batch_norm_sums': cr,
        'spike_sites': spike_sites, 'repeat_baseline_max_abs': repeat_max,
        'membrane_status': 'Not measured in this phase; post-reset values are not pre-fire voltages.',
        'optimizer_status': 'Actual optimizer groups and update trajectories require separate audit.',
        'elapsed_seconds': time.time() - started}
    if not args.smoke and any(row['samples'] != len(dataset) for row in results.values()):
        raise ValueError('Incomplete validation')
    np.savez_compressed(Path(args.output) / (args.job + '.logits.npz'),
                        **{name: value.numpy() for name, value in logits.items()})
    save(Path(args.output) / (args.job + '.json'), result)
    print('DIAGNOSTICS_COMPLETE ' + args.job, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--dataset', choices=('cifar100', 'dvs'), default='cifar100')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--server')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.job:
        plan = read(args.output / 'plan.json')
        if digest(__file__) != plan['evaluator_sha256']:
            raise ValueError('Evaluator changed')
        worker(args, plan)
        return
    records = [row for row in read(args.root / 'selection_frozen_before_test.json')
               if row['method'] == 'B' and ('_d8_' in row['name'] or args.dataset == 'dvs')]
    if args.smoke:
        records = records[:1]
    if not records:
        raise ValueError('No B checkpoints')
    gpus = list(map(int, args.gpus.split(',')))
    if len(gpus) != len(set(gpus)):
        raise ValueError('Duplicate GPU')
    for row in records:
        checkpoint = args.root / 'runs' / row['name'] / 'model_best.pth.tar'
        if digest(checkpoint) != row['checkpoint_sha256']:
            raise ValueError('Checkpoint mismatch')
        row['local_checkpoint'] = str(checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(root=str(args.root), source=str(args.source or args.root / 'source/src'),
                adapter=str(args.adapter or args.root), dataset=args.dataset, data=str(args.data),
                server=args.server, records=records, gpus=gpus, evaluator_sha256=digest(__file__),
                created_unix=time.time(), policy='Validation only; all four interventions; no test access.')
    save(args.output / 'plan.json', plan)

    def lane(index):
        gpu = gpus[index]
        for row in records[index::len(gpus)]:
            idle(gpu)
            prefix = str(Path(sys.executable).parent.parent)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', CUDA_PATH=prefix,
                       LD_LIBRARY_PATH=prefix + '/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
            command = [sys.executable, __file__, '--output', str(args.output), '--job', row['name']]
            if args.smoke:
                command.append('--smoke')
            with (args.output / (row['name'] + '.log')).open('x') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))
    save(args.output / 'COMPLETE.json', {'completed': [row['name'] for row in records], 'time': time.time()})


if __name__ == '__main__':
    main()
