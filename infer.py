"""
infer.py — command-line foley generation (no Gradio required)

Usage
-----
# Explicit prompt
python infer.py --video examples/hitting_a_plastic_bag.mp4 \\
                --prompt "hitting a plastic bag" \\
                [--theme cinematic] \\
                [--output output/my_run]

# Auto-caption with SmolVLM2 (no --prompt needed)
python infer.py --video examples/typing.mp4 --auto-caption
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
        default=None,
        metavar="TEXT",
        help='Sound description, e.g. "a dog barking". Required unless --auto-caption is set.',
    )
    parser.add_argument(
        "--auto-caption", "-a",
        action="store_true",
        help="Use SmolVLM2 to generate the prompt automatically from the video.",
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

    if not args.auto_caption and not args.prompt:
        parser.error("Provide --prompt or --auto-caption.")

    # Import here so --help is instant even before heavy libs are loaded
    from app.config import THEMES_CLI
    from app.pipeline import generate

    theme_key = THEMES_CLI.get(args.theme.lower())
    if theme_key is None:
        parser.error(
            f"Unknown theme '{args.theme}'. "
            f"Valid choices: {', '.join(THEMES_CLI)}"
        )

    prompt = args.prompt or ""
    if args.auto_caption:
        from app.caption import auto_caption
        print("Running auto-caption with SmolVLM2…")
        result = auto_caption(args.video)
        caption = result.get("value", "") if isinstance(result, dict) else str(result)
        if caption.startswith("Auto-caption failed"):
            parser.error(caption)
        print(f"Caption: {caption}")
        prompt = f"{prompt}, {caption}".strip(", ") if prompt else caption

    result = generate(
        video_path=args.video,
        prompt=prompt,
        theme=theme_key,
        output_dir=args.output,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
