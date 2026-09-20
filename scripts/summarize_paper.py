"""Recompute mean/sample SD and seed-paired contrasts from archived observations."""
import argparse
import json
from pathlib import Path
import statistics

DEFAULT = Path(__file__).resolve().parents[1] / 'paper_data/accuracy.json'


def summarize(path=DEFAULT):
    records = json.loads(Path(path).read_text(encoding='utf-8'))['records']
    indexed = {(row['config'], row['set'], row['method'], row['seed']): row for row in records}
    if len(indexed) != len(records) or len(records) != 140:
        raise ValueError('Expected 140 unique observations (100 internal, 40 official)')
    groups = sorted({(row['config'], row['set']) for row in records})
    if len(groups) != 7 or sum(row['set'] == 'H' for row in records) != 100:
        raise ValueError('Unexpected experiment matrix')
    output = []
    for config, partition in groups:
        values = {}
        for method in ('S', 'B', 'I', 'A0'):
            values[method] = [indexed[config, partition, method, seed]['accuracy'] for seed in range(42, 47)]
            output.append(dict(config=config, partition=partition, method=method,
                               mean=statistics.mean(values[method]), sd=statistics.stdev(values[method])))
        paired = [learned - fixed for learned, fixed in zip(values['B'], values['A0'])]
        output.append(dict(config=config, partition=partition, method='B-A0',
                           mean=statistics.mean(paired), sd=statistics.stdev(paired)))
    for row in records:
        if row['set'] == 'O':
            counterpart = indexed[row['config'], 'H', row['method'], row['seed']]
            if row['checkpoint_sha256'] != counterpart['checkpoint_sha256']:
                raise ValueError('Internal and official evaluations must use identical checkpoints')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print('| Configuration | Set | Method | Mean (pp) | Sample SD (pp) |')
    print('|---|---|---|---:|---:|')
    for row in summarize(args.input):
        print('| {config} | {partition} | {method} | {mean:.3f} | {sd:.3f} |'.format(**row))


if __name__ == '__main__':
    main()
