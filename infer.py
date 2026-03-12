"""
infer.py — command-line foley generation (no Gradio required)

Usage
-----
python infer.py --video examples/hitting_a_plastic_bag.mp4 \
                --prompt "hitting a plastic bag" \
                [--theme cinematic] \\
                [--output output/my_run]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate foley audio for a video from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        metavar="PATH",
        help="Path to the input video (.mp4 / .avi).",
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        metavar="TEXT",
        help='Sound description, e.g. "a dog barking".',
    )
    parser.add_argument(
        "--theme", "-t",
        default="none",
        metavar="THEME",
        help=(
            "Optional style theme. "
            "Choices: none, cinematic, cartoon, funny, "
            "horror, nature, sci-fi, fantasy, rock, jazz. "
            "(default: none)"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="DIR",
        help="Output directory. Defaults to gradio_output/run_<id>/.",
    )

    args = parser.parse_args()

    # Import here so --help is instant even before heavy libs are loaded
    from app.config import THEMES_CLI
    from app.pipeline import generate

    theme_key = THEMES_CLI.get(args.theme.lower())
    if theme_key is None:
        parser.error(
            f"Unknown theme '{args.theme}'. "
            f"Valid choices: {', '.join(THEMES_CLI)}"
        )

    result = generate(
        video_path=args.video,
        prompt=args.prompt,
        theme=theme_key,
        output_dir=args.output,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
