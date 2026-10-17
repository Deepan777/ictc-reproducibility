"""Create a new packet-study workspace with the recorded scientific parameters."""
import argparse
import importlib.util
from pathlib import Path
import os
import shutil
import sys

SOURCE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--external-inputs', type=Path, required=True)
    parser.add_argument('--native-runtime', type=Path, required=True)
    parser.add_argument('--historical-policy', type=Path, required=True,
        help='Recorded n13_selected_policy.json, regenerated or obtained separately; excluded from the code-only snapshot')
    args = parser.parse_args()
    root = args.work_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir()
    (root/'runtime').mkdir()
    (root/'audit').mkdir()
    shutil.copy2(SOURCE/'configs/revision_protocol.json', root/'configs/revision_protocol.json')
    policy = root/'code/configs/experiments/n13_selected_policy.json'
    policy.parent.mkdir(parents=True)
    shutil.copy2(args.historical_policy, policy)
    sys.path[:0] = [str(SOURCE/'code'), str(SOURCE/'code/src')]
    transform = load('topology_transform', SOURCE/'code/scripts/prepare_n13_transport_topologies.py')
    derived = root/'code/configs/experiments/n13_transport_derived'
    derived.mkdir()
    for filename, topology in (('Geant2012.graphml', 'geant-2012-03'), ('Garr201109.graphml', 'garr-2011-09')):
        transform.transform(args.external_inputs/filename, derived/f'{topology}.json', topology)
    for engine in ('ictc_n12_abilene', 'ictc_n13_transport'):
        shutil.copy2(args.native_runtime/f'{engine}.exe', root/f'runtime/ns3.46.1-{engine}.exe')
    os.environ['ICTC_REPRO_ROOT'] = str(root)
    import run_revision_native
    run_revision_native.prepare()
    print(root)


if __name__ == '__main__':
    main()
