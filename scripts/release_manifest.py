"""Refresh or verify the public release file hashes, excluding runtime caches."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'RELEASE_MANIFEST.json'


def files():
    excluded = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist'}
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ROOT.rglob('*')) if path.is_file() and path != MANIFEST
            and not excluded.intersection(path.relative_to(ROOT).parts)
            and path.suffix not in {'.pyc', '.pyo'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    actual = files()
    if args.write:
        payload = dict(release='0.2.0', distribution='code-and-aggregate-observations',
                       author='Zhiqi Cai', repository='https://github.com/OneDecisionC/SpikeBirkhoff', files=actual)
        MANIFEST.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    elif json.loads(MANIFEST.read_text(encoding='utf-8'))['files'] != actual:
        raise SystemExit('Release hashes differ; inspect changes before refreshing.')
    print('Verified file count:', len(actual))


if __name__ == '__main__':
    main()
