"""Independently reproduce the seed-block amendment from portable endpoint records.

Usage: python recompute_controlled.py --input-dir PATH --output PATH.json
Requires NumPy and SciPy. Refuses to overwrite its output; does not run controllers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Output already exists'
    records = json.loads((args.input_dir/'retained_endpoint_records.json').read_text())
    expected = json.loads((args.input_dir/'corrected_inference.json').read_text())
    reference = records['A0:default']
    seeds = expected['seeds']
    draws = np.random.default_rng(expected['bootstrap_seed']).integers(0, len(seeds), (10000, len(seeds)))
    output = {}
    for method, saved_metrics in expected['comparisons'].items():
        output[method] = {}
        assert set(records[method]) == set(reference)
        for metric, saved in saved_metrics.items():
            sums, counts, left, right = [], [], [], []
            for seed in seeds:
                pairs = [(reference[key][metric], records[method][key][metric])
                         for key in sorted(reference) if reference[key]['seed'] == seed]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                sums.append(float(np.array([a-b for a, b in pairs]).sum()))
                counts.append(len(pairs))
                left.extend(a for a, b in pairs)
                right.extend(b for a, b in pairs)
            sums, counts = np.array(sums), np.array(counts)
            denominators = counts[draws].sum(axis=1)
            assert (denominators > 0).all()
            samples = sums[draws].sum(axis=1)/denominators
            means = np.divide(sums, counts, out=np.zeros(len(seeds)), where=counts > 0)
            positive, negative = int((means > 1e-12).sum()), int((means < -1e-12).sum())
            p = float(binomtest(positive, positive+negative, .5).pvalue) if positive+negative else 1.
            result = dict(paired_count=int(counts.sum()), independent_blocks=int((counts > 0).sum()),
                          reference_mean=float(np.mean(left)), comparator_mean=float(np.mean(right)),
                          difference=float(sums.sum()/counts.sum()),
                          seed_bootstrap_95_ci=np.quantile(samples, [.025, .975]).tolist(),
                          exact_seed_sign_p=p, positive_blocks=positive, negative_blocks=negative,
                          zero_blocks=int((counts > 0).sum())-positive-negative,
                          paired_count_per_block=counts.tolist())
            output[method][metric] = result
    metrics = {metric for values in output.values() for metric in values}
    for metric in metrics:
        family = [(m, output[m][metric]['exact_seed_sign_p']) for m in (f'B{i}' for i in range(9))
                  if metric in output[m]]
        previous = 0.
        for i, (method, p) in enumerate(sorted(family, key=lambda row: row[1])):
            previous = max(previous, min(1., (len(family)-i)*p))
            output[method][metric]['holm_p'] = previous
    assert output == expected['comparisons'], 'Recomputed inference differs'
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(f'Exact numerical reproduction: {len(output)} comparison configurations')


if __name__ == '__main__':
    main()
