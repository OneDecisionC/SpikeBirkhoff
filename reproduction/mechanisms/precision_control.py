"""DVS last-mixer AMP/FP32 validation controls including mean readout and head spikes."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

from diagnose import digest, idle, read, save


def worker(args, plan):
    import random
    import numpy as np
    import torch
    sys.path[:0] = [plan['source'], plan['adapter']]
    from spikebirkhoff.config import load_config
    from spikebirkhoff.model import build_model
    from spikingjelly.clock_driven import functional
    from timm.models.helpers import clean_state_dict
    from rental_data import RecordingDataset
    assert plan['dataset'] == 'dvs'
    record = next(row for row in plan['records'] if row['name'] == args.job)
    checkpoint = Path(record['local_checkpoint'])
    assert digest(checkpoint) == record['checkpoint_sha256']
    cfg = load_config(checkpoint.parent / 'resolved.yaml', 'B')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_model(cfg, method='B', backend='cupy')
    initial = model.routing_layers[-1].mixing_logits.detach().clone()
    payload = torch.load(checkpoint, map_location='cpu')
    assert payload['epoch'] == record['validation_selected_epoch_zero_based']
    model.load_state_dict(clean_state_dict(payload['state_dict']), strict=True)
    del payload
    model.cuda().eval()
    router = model.routing_layers[-1]
    saved = router.mixing_logits.detach().clone()
    saved_mode = router.routing_mode
    buffers = {name: value.clone() for name, value in model.named_buffers()
               if 'running_' in name or 'num_batches_tracked' in name}
    dataset = RecordingDataset(plan['data'], Path(plan['root']) / 'split.json', 'validation')
    previous = read(args.plan.parent / (args.job + '.json'))
    assert previous['split_sha256'] == digest(Path(plan['root']) / 'split.json')
    for item in dataset.records:
        assert digest(dataset.root / item['relative']) == item['sha256']
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.get('val_batch_size', cfg['batch_size']),
                                         shuffle=False, num_workers=4)
    captured = {}

    def mean_hook(module, inputs, output):
        captured['mean_readout'] = output.mean(0).detach().float().cpu()

    def head_pre(module, inputs):
        captured['head_input'] = inputs[0].detach().float().cpu()

    def head_post(module, inputs, output):
        captured['head_spikes'] = output.detach().float().cpu()

    handles = [router.register_forward_hook(mean_hook), model.head_lif.register_forward_pre_hook(head_pre),
               model.head_lif.register_forward_hook(head_post)]
    variants = ('B', 'last_A0', 'last_I')
    precisions = ('AMP', 'FP32')
    tensors = {}
    predictions = {precision + '_' + variant: [] for precision in precisions for variant in variants}
    labels_all = []

    def compare(name, current, reference):
        totals = tensors.setdefault(name, {})
        for site, value in current.items():
            baseline = reference[site]
            difference = value.double() - baseline.double()
            stats = totals.setdefault(site, {'count': 0, 'changed': 0, 'max_abs': 0., 'delta_square_sum': 0., 'reference_square_sum': 0.})
            stats['count'] += value.numel()
            stats['changed'] += int(value.ne(baseline).sum())
            stats['max_abs'] = max(stats['max_abs'], float(difference.abs().max()))
            stats['delta_square_sum'] += float(difference.square().sum())
            stats['reference_square_sum'] += float(baseline.double().square().sum())

    def forward(inputs, precision):
        captured.clear()
        functional.reset_net(model)
        try:
            inputs = inputs.float().cuda()
            if precision == 'AMP':
                inputs = inputs.half()
            with torch.cuda.amp.autocast(enabled=precision == 'AMP'):
                output = model(inputs)[0]
            if cfg.get('TET', False):
                output = output.mean(0)
            assert torch.isfinite(output).all()
            captured['logits'] = output.float().cpu()
            return dict(captured)
        finally:
            functional.reset_net(model)

    started = time.time()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(loader):
            labels_all.append(targets)
            amp_baseline = None
            for precision in precisions:
                baseline = None
                for variant in variants:
                    router.mixing_logits.copy_(saved)
                    router.routing_mode = saved_mode
                    if variant == 'last_A0':
                        router.mixing_logits.copy_(initial.to(saved))
                    elif variant == 'last_I':
                        router.routing_mode = 'fixed_I'
                    current = forward(inputs, precision)
                    if variant == 'B':
                        baseline = current
                        repeated = forward(inputs, precision)
                        assert all(torch.equal(value, repeated[site]) for site, value in current.items())
                        if precision == 'AMP':
                            amp_baseline = current
                        else:
                            compare('FP32_B_vs_AMP_B', current, amp_baseline)
                    compare(precision + '_' + variant, current, baseline)
                    predictions[precision + '_' + variant].append(current['logits'])
            if batch % 10 == 0:
                print('PRECISION_BATCH ' + str(batch), flush=True)
            if args.smoke:
                break
    labels = torch.cat(labels_all)
    metrics = {}
    arrays = {}
    for name, outputs in predictions.items():
        output = torch.cat(outputs)
        reference = torch.cat(predictions[name.split('_')[0] + '_B'])
        arrays[name] = output.numpy()
        metrics[name] = {'samples': len(labels), 'accuracy_percent': 100 * int(output.argmax(1).eq(labels).sum()) / len(labels),
                         'prediction_disagreement_percent': 100 * int(output.argmax(1).ne(reference.argmax(1)).sum()) / len(labels)}
    if not args.smoke:
        assert len(labels) == 1000
        assert abs(metrics['AMP_B']['accuracy_percent'] - previous['metrics']['B']['accuracy']) < .0001
    for name, value in model.named_buffers():
        if name in buffers:
            assert torch.equal(buffers[name], value)
    for handle in handles:
        handle.remove()
    np.savez_compressed(args.output / (args.job + '.logits.npz'), labels=labels.numpy(), **arrays)
    save(args.output / (args.job + '.json'), {'server': plan['server'], 'seed': cfg['seed'],
        'checkpoint_absolute_path': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
        'result_absolute_path': str(args.output / (args.job + '.json')), 'script_sha256': digest(__file__),
        'partition': 'validation', 'smoke_only': args.smoke, 'metrics': metrics, 'paired_tensors': tensors,
        'baseline_repeated_identical_every_batch': True, 'elapsed_seconds': time.time() - started,
        'scope': 'Same AMP-trained checkpoint; evaluation precision changes only, not FP32 retraining.'})
    print('PRECISION_COMPLETE ' + args.job, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--job')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    if args.job:
        worker(args, plan)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    records = plan['records'][:1] if args.smoke else plan['records']
    gpus = list(map(int, args.gpus.split(',')))
    save(args.output / 'precision_plan.json', {'records': records, 'prior_plan': str(args.plan),
         'script_sha256': digest(__file__), 'gpus': gpus, 'smoke_only': args.smoke,
         'policy': 'All five seeds, AMP/FP32 x B/last_A0/last_I, original validation only.'})

    def lane(index):
        for record in records[index::len(gpus)]:
            idle(gpus[index])
            prefix = str(Path(sys.executable).parent.parent)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[index]), OMP_NUM_THREADS='4',
                MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', CUDA_PATH=prefix,
                LD_LIBRARY_PATH=prefix + '/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
            command = [sys.executable, __file__, '--plan', str(args.plan), '--output', str(args.output), '--job', record['name']]
            if args.smoke:
                command.append('--smoke')
            with (args.output / (record['name'] + '.log')).open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))
    save(args.output / 'COMPLETE.json', {'completed': [record['name'] for record in records]})


if __name__ == '__main__':
    main()
