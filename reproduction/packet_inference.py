"""Paired seed-block inference with the reported matched and adaptation families."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/src')]
from analyze_revision import summary


def adjust(family, endpoint):
    ordered = sorted(family, key=lambda name: family[name][endpoint]['exact_seed_block_sign_p_two_sided'])
    previous = 0.
    for rank, name in enumerate(ordered):
        p = family[name][endpoint]['exact_seed_block_sign_p_two_sided']
        previous = max(previous, min(1., (len(ordered)-rank)*p))
        family[name][endpoint]['holm_adjusted_p'] = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.records.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['seed'] = int(row['seed'])
        for field in ('harmful', 'beneficial', 'change', 'opportunity', 'retained'):
            row[field] = row[field] in ('True', '1')
    seeds = sorted({row['seed'] for row in rows})
    if len(seeds) != 24:
        raise ValueError('Independent packet population requires 24 seeds')
    for method in {row['method'] for row in rows}:
        if any(sum(row['method'] == method and row['seed'] == seed for row in rows) != 12 for seed in seeds):
            raise ValueError('Every method/seed block must contain all 12 condition cells')
    matched = {}
    for proposer in ('Learned', 'Fluid'):
        admitted = proposer+'ICTC'
        result = summary([row for row in rows if row['method'] in (proposer, admitted)], baseline=proposer)
        matched[proposer] = {endpoint: result['methods'][admitted][endpoint+'_minus_'+proposer]
                             for endpoint in ('harmful', 'utility')}
    adaptation = {}
    for method in ('ThresholdICTC', 'AdaptedICTC3', 'AdaptedICTC6', 'AdaptedICTC12'):
        result = summary([row for row in rows if row['method'] in (method, 'OriginalICTCObservable')],
                         baseline='OriginalICTCObservable')
        adaptation[method] = {endpoint: result['methods'][method][endpoint+'_minus_OriginalICTCObservable']
                              for endpoint in ('harmful', 'utility')}
    for endpoint in ('harmful', 'utility'):
        adjust(matched, endpoint)
        adjust(adaptation, endpoint)
    result = {'independent_unit': 'complete seed, all 12 condition cells and paired methods',
              'seeds': seeds, 'bootstrap_resamples': 10000, 'bootstrap_seed': 20261014,
              'confidence_intervals': 'nominal percentile seed bootstrap',
              'tests': 'two-sided exact sign tests on nonzero seed-mean differences',
              'multiplicity': 'Holm by endpoint within two matched contrasts or four adaptation contrasts',
              'matched': matched, 'adaptation': adaptation}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
