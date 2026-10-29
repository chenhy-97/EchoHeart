"""Stage 3 validate entry point."""

from stage2.training import main


if __name__ == "__main__":
    main(stage=3, mode="validate")
