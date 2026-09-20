"""Validate cached recordings and create the frozen 8:1:1 split."""
import argparse
from pathlib import Path
from rental_data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing split')
    prepare(args.data, args.output, 20260919)


if __name__ == '__main__':
    main()
