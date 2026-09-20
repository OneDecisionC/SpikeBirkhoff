"""Predeclared equal-magnitude, center-directed mixer perturbations."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from diagnose import digest, idle, read, save

EDGES = [-2, -1, -.5, -.1, -.05, -.01, -.001, 0, .001, .01, .05, .1, .5, 1, 2]
ABS_EDGES = [.001, .01, .05, .1, .5, 1]


def worker(args, plan):
    import random
    import numpy as np
    import torch
    sys.path[:0] = [plan['source'], plan['adapter']]
    from spikebirkhoff.config import load_config
    from spikebirkhoff.model import build_model
    from timm.models.helpers import clean_state_dict
    from spikingjelly.clock_driven import functional, neuron, neuron_kernel
    from spikingjelly import configure
    row = next(record for record in plan['records'] if record['name'] == args.job)
    checkpoint = Path(row['local_checkpoint'])
    assert digest(checkpoint) == row['checkpoint_sha256']
    cfg = load_config(checkpoint.parent / 'resolved.yaml', 'B')
    if args.precision == 'fp32':
        cfg['amp'] = False
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_model(cfg, method='B', backend='cupy')
    initial = [router.mixing_logits.detach().clone() for router in model.routing_layers]
    payload = torch.load(checkpoint, map_location='cpu')
    assert payload['epoch'] == row['validation_selected_epoch_zero_based']
    model.load_state_dict(clean_state_dict(payload['state_dict']), strict=True)
    del payload
    model.cuda().eval()
    original = [{name: value.detach().clone() for name, value in router.state_dict().items()}
                for router in model.routing_layers]
    modes = [router.routing_mode for router in model.routing_layers]
    buffers = {name: value.clone() for name, value in model.named_buffers()
               if 'running_' in name or 'num_batches_tracked' in name}
    previous = read(args.plan.parent / (args.job + '.json'))
    assert previous['checkpoint_sha256'] == digest(checkpoint)
    assert previous['split_sha256'] == digest(Path(plan['root']) / 'split.json')
    if plan['dataset'] == 'cifar100':
        from holdout import load_partition
        from timm.data import create_loader, resolve_data_config
        dataset = load_partition(plan['root'], 'validation')
        data_cfg = resolve_data_config(cfg, model=model)
        full_loader = create_loader(dataset, input_size=data_cfg['input_size'], batch_size=cfg.get('val_batch_size', 64), is_training=False,
            use_prefetcher=False, interpolation=data_cfg['interpolation'], mean=data_cfg['mean'],
            std=data_cfg['std'], crop_pct=data_cfg['crop_pct'], num_workers=0, persistent_workers=False)
    else:
        from rental_data import RecordingDataset
        dataset = RecordingDataset(plan['data'], Path(plan['root']) / 'split.json', 'validation')
        for item in dataset.records:
            assert digest(dataset.root / item['relative']) == item['sha256']
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, previous['probe_indices']),
                                        batch_size=8, shuffle=False, num_workers=2)
    if plan['dataset'] != 'cifar100':
        full_loader = torch.utils.data.DataLoader(dataset,
            batch_size=cfg.get('val_batch_size', cfg['batch_size']), shuffle=False, num_workers=4)
    variants = ('B',) + tuple(f'layer_{index}_center' for index in range(len(model.routing_layers)))
    perturbations = {}
    for index, router in enumerate(model.routing_layers):
        assert router.routing_mode == 'birkhoff' and router.mixing_logits.shape == (2,)
        alpha = float(router.mixing_matrix().detach().float()[0, 0])
        target = alpha - .01 if alpha >= .5 else alpha + .01
        assert 0 < target < 1
        original_matrix = router.mixing_matrix().detach().float().clone()
        with torch.no_grad():
            router.mixing_logits.copy_(router.mixing_logits.new_tensor([target, 1 - target]).log())
            new_matrix = router.mixing_matrix().detach().float().clone()
            achieved = float(new_matrix[0, 0]) - alpha
            assert abs(abs(achieved) - .01) < 1e-6
            assert float((new_matrix.sum(0) - 1).abs().max()) < 1e-6
            assert float((new_matrix.sum(1) - 1).abs().max()) < 1e-6
            assert float(new_matrix.min()) >= 0
            with torch.cuda.amp.autocast(enabled=cfg.get('amp', False)):
                new_executed = router.mixing_matrix().detach().float().clone()
            router.load_state_dict(original[index], strict=True)
            with torch.cuda.amp.autocast(enabled=cfg.get('amp', False)):
                original_executed = router.mixing_matrix().detach().float().clone()
        perturbations[str(index)] = {
            'alpha_original_fp32': alpha, 'alpha_target': target, 'achieved_delta_fp32': achieved,
            'matrix_delta_fro_fp32': float((new_matrix - original_matrix).norm()),
            'matrix_delta_fro_evaluation_precision': float((new_executed - original_executed).norm()),
            'new_column_sum_error_evaluation_precision': float((new_executed.sum(0) - 1).abs().max()),
            'direction_rule': 'toward 0.5; at 0.5 choose negative'}


    def intervene(variant):
        for index, router in enumerate(model.routing_layers):
            router.load_state_dict(original[index], strict=True)
            router.routing_mode = modes[index]
        if variant != 'B':
            _, target, mode = variant.split('_')
            router = model.routing_layers[int(target)]
            alpha = perturbations[target]['alpha_target']
            router.mixing_logits.copy_(router.mixing_logits.new_tensor([alpha, 1 - alpha]).log())

    measurements = {variant: {} for variant in variants}
    block_input_change = {variant: {} for variant in variants}
    context = {'enabled': False, 'site': None, 'variant': None, 'samples': {}}
    for index, router in enumerate(model.routing_layers):
        native_read = router.read

        def tracked_read(streams, index=index, native_read=native_read):
            output = native_read(streams)
            if context['enabled']:
                value = output.detach().float().cpu()
                if context['variant'] == 'B':
                    context['baseline_inputs'][index] = value
                baseline = context['baseline_inputs'][index]
                totals = block_input_change[context['variant']].setdefault(str(index), [0.0, 0.0])
                totals[0] += float((value - baseline).double().square().sum())
                totals[1] += float(baseline.double().square().sum())
            return output

        router.read = tracked_read
    edges = torch.tensor(EDGES, device='cuda')
    original_apply = neuron_kernel.MultiStepLIFNodePTT.apply

    class CaptureContext:
        def save_for_backward(self, *tensors):
            self.pre_fire = tensors[0]

    def observed_apply(inputs, voltage, decay_input, tau, threshold, reset, detach_reset, surrogate):
        if not context['enabled'] or context['site'] is None:
            return original_apply(inputs, voltage, decay_input, tau, threshold, reset, detach_reset, surrogate)
        capture = CaptureContext()
        cache_setting = configure.save_spike_as_bool_in_neuron_kernel
        configure.save_spike_as_bool_in_neuron_kernel = False
        try:
            spikes, voltages = neuron_kernel.MultiStepLIFNodePTT.forward(capture,
                inputs.detach().requires_grad_(True), voltage.detach(), decay_input, tau,
                threshold, reset, detach_reset, surrogate)
        finally:
            configure.save_spike_as_bool_in_neuron_kernel = cache_setting
        pre_fire = capture.pre_fire[..., :inputs.shape[-1]]
        if not torch.isfinite(pre_fire).all():
            raise ValueError('Non-finite pre-fire membrane')
        if not torch.equal(pre_fire.ge(threshold), spikes.bool()):
            raise ValueError('Native pre-fire threshold does not reproduce spikes')
        margin = (pre_fire.float() - threshold) / abs(threshold)
        histogram = torch.bincount(torch.bucketize(margin.flatten(), edges), minlength=len(EDGES) + 1).cpu()
        site = context['site']
        result = measurements[context['variant']].setdefault(site,
            {'histogram': [0] * (len(EDGES) + 1), 'count': 0, 'margin_sum': 0.0,
             'margin_square_sum': 0.0, 'threshold': threshold, 'dtype': str(inputs.dtype),
             'threshold_spike_mismatches': 0, 'paired_count_by_baseline_abs_margin': [0] * 7,
             'paired_spike_flips_by_baseline_abs_margin': [0] * 7,
             'paired_margin_delta_square_sum': 0.0})
        result['histogram'] = [left + right for left, right in zip(result['histogram'], histogram.tolist())]
        result['count'] += margin.numel()
        result['margin_sum'] += float(margin.double().sum())
        result['margin_square_sum'] += float(margin.double().square().sum())
        stride = max(1, margin.numel() // 8192)
        sampled_margin = margin.flatten()[::stride][:8192].cpu()
        sampled_spikes = spikes.flatten()[::stride][:8192].bool().cpu()
        if context['variant'] == 'B':
            context['samples'][site] = (sampled_margin, sampled_spikes)
        baseline_margin, baseline_spikes = context['samples'][site]
        bucket = torch.bucketize(baseline_margin.abs(), torch.tensor(ABS_EDGES))
        counts = torch.bincount(bucket, minlength=7)
        flips = torch.bincount(bucket[sampled_spikes != baseline_spikes], minlength=7)
        for index in range(7):
            result['paired_count_by_baseline_abs_margin'][index] += int(counts[index])
            result['paired_spike_flips_by_baseline_abs_margin'][index] += int(flips[index])
        result['paired_margin_delta_square_sum'] += float((sampled_margin - baseline_margin).double().square().sum())
        return spikes, voltages

    handles = []
    sites = []
    for name, module in model.named_modules():
        if name in previous['spike_sites']:
            if type(module) is not neuron.MultiStepLIFNode or module.backend != 'cupy':
                raise ValueError('Unsupported neuron type; no approximate replacement')
            sites.append(name)

            def entering(module, inputs, name=name):
                context['site'] = name

            def leaving(module, inputs, output):
                context['site'] = None

            handles.append(module.register_forward_pre_hook(entering))
            handles.append(module.register_forward_hook(leaving))
    assert sites and set(sites) == set(previous['spike_sites'])
    neuron_kernel.MultiStepLIFNodePTT.apply = observed_apply

    def forward(inputs, enabled):
        context['enabled'] = enabled
        functional.reset_net(model)
        try:
            inputs = inputs.float().cuda()
            if cfg.get('amp', False):
                inputs = inputs.half()
            with torch.cuda.amp.autocast(enabled=cfg.get('amp', False)):
                result = model(inputs)[0]
            return result.detach().clone()
        finally:
            functional.reset_net(model)

    started = time.time()
    samples = 0
    validation = {}
    logits = {}
    with torch.no_grad():
        for variant in variants:
            intervene(variant)
            outputs, labels_all = [], []
            for batch, (inputs, labels) in enumerate(full_loader):
                output = forward(inputs, False)
                if cfg.get('TET', False):
                    output = output.mean(0)
                assert torch.isfinite(output).all()
                outputs.append(output.float().cpu())
                labels_all.append(labels)
                if args.smoke:
                    break
            output = torch.cat(outputs)
            targets = torch.cat(labels_all)
            logits[variant] = output.numpy()
            if variant == 'B':
                baseline = output
            validation[variant] = {
                'samples': len(targets),
                'accuracy_percent': 100 * int(output.argmax(1).eq(targets).sum()) / len(targets),
                'prediction_disagreement_percent': 100 * int(output.argmax(1).ne(baseline.argmax(1)).sum()) / len(targets),
                'relative_logit_fro': float((output - baseline).norm() / baseline.norm().clamp_min(1e-12)),
                'mean_abs_probability_change': float((output.softmax(1) - baseline.softmax(1)).abs().mean())}
            if not args.smoke:
                assert len(targets) == len(dataset)
            print('VALIDATION ' + variant + ' ' + json.dumps(validation[variant]), flush=True)
        np.savez_compressed(args.output / (args.job + '.logits.npz'),
                            targets=targets.numpy(), **logits)
        for batch, (inputs, labels) in enumerate(loader):
            context['samples'] = {}
            context['baseline_inputs'] = {}
            for variant in variants:
                context['variant'] = variant
                intervene(variant)
                reference = forward(inputs, False)
                observed = forward(inputs, True)
                if not torch.equal(reference, observed):
                    raise ValueError('Instrumentation changed native logits')
            samples += len(labels)
            if batch % 10 == 0:
                print('PROBE_BATCH ' + str(batch), flush=True)
            if args.smoke:
                break
    for name, value in model.named_buffers():
        if name in buffers and not torch.equal(buffers[name], value):
            raise ValueError('BN changed')
    for handle in handles:
        handle.remove()
    neuron_kernel.MultiStepLIFNodePTT.apply = original_apply
    if not args.smoke:
        assert samples == len(previous['probe_indices'])
    save(args.output / (args.job + '.json'), {'server': plan['server'], 'seed': cfg['seed'],
        'checkpoint_absolute_path': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
        'result_absolute_path': str(args.output / (args.job + '.json')), 'samples': samples,
        'partition': 'validation_and_fixed_validation_probe', 'smoke_only': args.smoke,
        'validation': validation, 'variants': variants,
        'precision_requested': args.precision, 'amp_effective': cfg.get('amp', False),
        'perturbations': perturbations, 'target_abs_alpha_delta': .01,
        'block_input_delta_and_reference_squared_sums': block_input_change,
        'layer_indexing': 'zero-based; final layer is a structural negative control',
        'probe_indices': previous['probe_indices'][:samples], 'batch_size': 8,
        'signed_margin_histogram_interior_edges': EDGES, 'outer_edges': ['-infinity', '+infinity'],
        'paired_abs_margin_interior_edges': ABS_EDGES, 'paired_max_positions_per_site_batch': 8192,
        'margin_definition': '(native_h_seq - threshold) / abs(threshold)',
        'method': 'One mixer changed at a time; unchanged native CuPy forward and h_seq capture, no backward.',
        'instrumented_vs_native_logits_exact_equal_every_batch_variant': True,
        'native_kernel_source_sha256': digest(neuron_kernel.__file__),
        'script_sha256': digest(__file__), 'prior_result_sha256': digest(args.plan.parent / (args.job + '.json')),
        'measurements': measurements, 'elapsed_seconds': time.time() - started})
    print('EQUAL_PERTURBATION_COMPLETE ' + args.job, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--job')
    parser.add_argument('--precision', choices=('native', 'fp32'), default='native')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    if args.job:
        worker(args, plan)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    records = plan['records'][:1] if args.smoke else plan['records']
    gpus = list(map(int, args.gpus.split(',')))
    assert len(gpus) == len(set(gpus))
    save(args.output / 'equal_perturbation_plan.json', {'prior_plan': str(args.plan), 'prior_plan_sha256': digest(args.plan),
        'script_sha256': digest(__file__), 'records': records, 'gpus': gpus, 'smoke_only': args.smoke,
        'precision': args.precision, 'abs_alpha_delta': .01,
        'policy': 'Fixed abs(alpha change)=0.01 toward 0.5, all layers and final negative control; validation only.'})

    def lane(index):
        for row in records[index::len(gpus)]:
            idle(gpus[index])
            prefix = str(Path(sys.executable).parent.parent)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[index]), OMP_NUM_THREADS='4',
                MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', CUDA_PATH=prefix,
                LD_LIBRARY_PATH=prefix + '/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
            command = [sys.executable, __file__, '--plan', str(args.plan), '--output', str(args.output), '--job', row['name'], '--precision', args.precision]
            if args.smoke:
                command.append('--smoke')
            with (args.output / (row['name'] + '.log')).open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))
    save(args.output / 'COMPLETE.json', {'completed': [row['name'] for row in records]})


if __name__ == '__main__':
    main()
