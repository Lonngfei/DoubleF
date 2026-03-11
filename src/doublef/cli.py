#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from . import __author__, __description__, __program__, __version__
from .main import run_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doublef",
        description=__description__,
        add_help=True,
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to config file, for example: example.config",
    )
    return parser


def print_banner() -> None:
    print(f'{__program__}: {__description__}')
    print(f"Version: {__version__}")
    print("Usage: doublef example.config")

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config is None:
        print_banner()
        print()
        return

    config_file = args.config

    if not os.path.isfile(config_file):
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    run_from_config(config_file)


if __name__ == "__main__":
    main()