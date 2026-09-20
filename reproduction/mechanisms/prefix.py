"""Five-epoch prefixes with unchanged full-run scheduler and actual step logging."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import types

from diagnose import digest, idle, read, save


class PrefixComplete(Exception):
    pass


def run(args, plan):
    sys.path[:0] = [plan['source'], plan['adapter']]
    import torch
    from spikebirkhoff import train, training_engine
    row = next(record for record in plan['records'] if record['name'] == args.job)
    if plan['dataset'] == 'cifar100':
        from holdout import load_partition
        training = load_partition(plan['root'], 'train')
        validation = load_partition(plan['root'], 'validation')

        def dataset(name, root, split, is_training, **kwargs):
            if (split, is_training) == ('train', True):
                return training
            if (split, is_training) == ('validation', False):
                return validation
            raise ValueError('Forbidden partition request')

        training_engine.create_dataset = dataset
    destination = args.output / args.job
    trace = (args.output / (args.job + '.updates.jsonl')).open('x')
    state = {'epoch': None, 'steps': 0}
    original_create = training_engine.create_optimizer_v2

    def create(model, *values, **kwargs):
        optimizer = original_create(model, *values, **kwargs)
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        routes = [(names[id(parameter)], parameter, group) for group in optimizer.param_groups
                  for parameter in group['params'] if names[id(parameter)].startswith('routing_layers.')]
        save(args.output / (args.job + '.groups.json'),
             [{'name': name, 'weight_decay': group['weight_decay'], 'requires_grad': parameter.requires_grad}
              for name, parameter, group in routes])
        original_step = optimizer.step

        def step(optimizer_self, *step_args, **step_kwargs):
            before = {name: parameter.detach().clone() for name, parameter, group in routes}
            matrices = [router.mixing_matrix().detach().float().clone() for router in model.routing_layers]
            gradients = {name: None if parameter.grad is None else float(parameter.grad.detach().float().norm())
                         for name, parameter, group in routes}
            outcome = original_step(*step_args, **step_kwargs)
            state['steps'] += 1
            record = {'epoch': state['epoch'], 'actual_optimizer_step': state['steps'],
                'route_parameters': [{'name': name, 'gradient_norm_after_unscale_and_clip': gradients[name],
                    'parameter_delta_norm': float((parameter.detach() - before[name]).norm()),
                    'lr': group['lr'], 'weight_decay': group['weight_decay']}
                    for name, parameter, group in routes],
                'layers': [{'layer': index,
                    'matrix_delta_norm': float((router.mixing_matrix().detach().float() - matrices[index]).norm()),
                    'read_weights': router.read_weights.detach().cpu().tolist(),
                    'write_weights': router.write_weights.detach().cpu().tolist(),
                    'matrix': router.mixing_matrix().detach().float().cpu().tolist()}
                    for index, router in enumerate(model.routing_layers)]}
            trace.write(json.dumps(record, allow_nan=False) + '\n')
            if state['steps'] % 50 == 0:
                trace.flush()
            return outcome

        optimizer.step = types.MethodType(step, optimizer)
        return optimizer

    training_engine.create_optimizer_v2 = create
    original_epoch = training_engine.train_one_epoch

    def epoch(epoch, *values, **kwargs):
        if epoch >= (1 if args.smoke else 5):
            raise PrefixComplete()
        state['epoch'] = epoch
        return original_epoch(epoch, *values, **kwargs)

    training_engine.train_one_epoch = epoch
    config = Path(row['local_checkpoint']).parent / 'resolved.yaml'
    command = ['--config', str(config), '--method', 'B', '--output', str(args.output),
               '--experiment', args.job, '--set', 'collect_diagnostics=true']
    if plan['dataset'] == 'dvs':
        command += ['--data-dir', plan['data'], '--set', 'split_manifest=' + str(Path(plan['root']) / 'split.json')]
    if args.smoke:
        command += ['--set', 'max_train_batches=2', '--set', 'max_val_batches=2']
    try:
        train.main(command)
        raise RuntimeError('Full scheduler unexpectedly ended before prefix stop')
    except PrefixComplete:
        metadata = read(destination / 'run.json')
        metadata['status'] = 'diagnostic_prefix_complete'
        (destination / 'run.json').write_text(json.dumps(metadata, indent=2))
        save(args.output / (args.job + '.complete.json'),
             {'server': plan['server'], 'run_absolute_path': str(destination), 'state': state,
              'full_scheduler_retained': True, 'epochs_completed': 1 if args.smoke else 5,
              'smoke_only': args.smoke, 'test_accessed': False, 'script_sha256': digest(__file__),
              'limitation': 'Only actual optimizer steps logged; AMP skipped steps do not call optimizer.step.'})
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--job')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    if args.job:
        run(args, plan)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    records = plan['records'][:1] if args.smoke else plan['records']
    gpus = list(map(int, args.gpus.split(',')))
    save(args.output / 'prefix_plan.json', {'evaluation_plan': str(args.plan), 'epochs': 5,
        'scheduler_epochs_unchanged': True, 'smoke_only': args.smoke, 'script_sha256': digest(__file__),
        'records': records, 'gpus': gpus})

    def lane(index):
        for row in records[index::len(gpus)]:
            idle(gpus[index])
            prefix = str(Path(sys.executable).parent.parent)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[index]), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                       CUDA_PATH=prefix, LD_LIBRARY_PATH=prefix + '/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
            command = [sys.executable, __file__, '--plan', str(args.plan), '--output', str(args.output),
                       '--job', row['name']]
            if args.smoke:
                command.append('--smoke')
            with (args.output / (row['name'] + '.log')).open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))
    save(args.output / 'COMPLETE.json', {'completed': [row['name'] for row in records]})


if __name__ == '__main__':
    main()
