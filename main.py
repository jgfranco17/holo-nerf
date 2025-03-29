# Base libraries
import os
import time
from argparse import ArgumentParser

# NeRF library
import holonerf


def config() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="The source directory to get array data from",
    )


if __name__ == "__main__":
    parser = config()
    args = parser.parse_args()
