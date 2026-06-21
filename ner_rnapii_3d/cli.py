"""Command-line interface for the 1D simulation engine.

    ner-sim run CONFIG.yaml [--output DIR] [--seed N]

Runs the 1D loop-extrusion (+ RNAPII + lesion) dynamics for the ``lef`` section
of CONFIG.yaml and writes ``LEFPositions.h5`` (to ``--output`` if given, else to
the config's ``output_path``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_lef_config
from .engine.simulate import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ner-sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the 1D simulation and write LEFPositions.h5")
    r.add_argument("config", type=Path, help="pipeline/lef YAML config")
    r.add_argument("--output", "-o", type=Path, default=None,
                   help="output directory (LEFPositions.h5 is written inside it)")
    r.add_argument("--seed", type=int, default=None,
                   help="override the config RNG seed")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cfg = load_lef_config(args.config, output_path=args.output)
        if args.seed is not None:
            cfg.seed = args.seed
        out = run(cfg)
        print(f"wrote {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
