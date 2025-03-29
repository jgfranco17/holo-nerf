import argparse

from holonerf.mapping import DepthMap


def load_args() -> argparse.Namespace:
    """Load the arguments from the command line.

    Returns:
        Namespace: AArgs data object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="samples/depthdata",
        help="Path to dataset",
    )
    parser.add_argument(
        "--frame-rate", "-r", type=int, default=15, help="Output video frame rate"
    )
    parser.add_argument(
        "--color", "-c", type=str, default="hot", help="Colormap styling"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    depthmap = DepthMap(directory=args.source, color=args.color)
    depthmap.display(scale=0.8)
