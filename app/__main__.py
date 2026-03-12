"""
Entry point for `python -m app`.

Usage
-----
    conda activate foley
    python -m app                    # default port 7860
    python -m app --port 8080
    python -m app --share            # Gradio public tunnel
"""

import argparse
import os

from .config import OUTPUT_BASE
from .ui import build_ui


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FoleyADeux — AI foley sound generation web UI"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio tunnel URL",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
