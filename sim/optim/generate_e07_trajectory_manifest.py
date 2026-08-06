"""CLI for generating E07 trajectory manifests."""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optim.e07_trajectory_manifest import (
    DEFAULT_E07_MANIFEST,
    DEFAULT_E07_V2_CANDIDATE_MANIFEST,
    build_e07_param228_specs,
    build_e07_tiny_specs,
    build_e07_v2_candidate_specs,
    write_e07_manifest,
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate E07 trajectory manifest')
    parser.add_argument('--preset', default='param228',
                        choices=['param228', 'tiny', 'v2-candidates'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if args.preset == 'param228':
        specs = build_e07_param228_specs(seed=args.seed)
        output = args.output or DEFAULT_E07_MANIFEST
    elif args.preset == 'v2-candidates':
        specs = build_e07_v2_candidate_specs(seed=args.seed)
        output = args.output or DEFAULT_E07_V2_CANDIDATE_MANIFEST
    else:
        specs = build_e07_tiny_specs()
        output = args.output or DEFAULT_E07_MANIFEST

    summary = write_e07_manifest(
        specs, output, preset=args.preset, seed=args.seed)
    print(f'manifest: {output}')
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


if __name__ == '__main__':
    main()
