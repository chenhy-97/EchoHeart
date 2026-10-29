"""Evaluate the final Stage 3 model at study level."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2.training import _parser, validate


def main():
    parser = _parser(stage=3, mode="validate")
    parser.description = __doc__
    parser.set_defaults(split="test")
    validate(stage=3, args=parser.parse_args())


if __name__ == "__main__":
    main()
