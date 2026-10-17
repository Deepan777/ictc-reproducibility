"""Acquire external inputs separately; verify the fixed content hashes before use."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

INPUTS = {
    'Abilene.graphml': ('https://topology-zoo.org/files/Abilene.graphml',
        '8CD694280D98B336BB9B51FC3B2129A514F1B1B1AC80F2022ACA02A57EF1E371'),
    'Geant2012.graphml': ('https://topology-zoo.org/files/Geant2012.graphml',
        '8487B2FB091D53F1317035162DF0D8EF436C9E1D10727E26865B0ECD89D4786E'),
    'Garr201109.graphml': ('https://topology-zoo.org/files/Garr201109.graphml',
        'A41B22107ABD9900C84B3B3D9F64F141981F134701A1C67B95EA031A29C8C318'),
    'mawi-202004221400.html': ('https://mawi.wide.ad.jp/mawi/samplepoint-G/2020/202004221400.html',
        '5823E9208FC97E878929494EEA201E15D27221CB590DCC1BFF6A0E51C5D5FE07'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in INPUTS.items():
        path = args.directory/name
        if not path.exists():
            if args.verify_only:
                raise FileNotFoundError(path)
            request = urllib.request.Request(url, headers={'User-Agent': 'ICTC-reproduction'})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            actual = hashlib.sha256(content).hexdigest().upper()
            if actual != expected:
                raise ValueError(f'External source bytes changed: {name}: {actual}')
            with path.open('xb') as stream:
                stream.write(content)
        if hashlib.sha256(path.read_bytes()).hexdigest().upper() != expected:
            raise ValueError(f'Input hash mismatch: {name}')
        print(f'Verified {name}')


if __name__ == '__main__':
    main()
