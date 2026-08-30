"""
Generate TTS audio from markdown content files using Microsoft Edge TTS (free).
Usage:
  python generate_audio.py                        # Generate all missing audio
  python generate_audio.py --module neuro_boards  # Single module
  python generate_audio.py --concept "GBM"        # Single concept (partial match)
  python generate_audio.py --dry-run              # Show what would be generated
  python generate_audio.py --voice en-US-GuyNeural  # Different voice
"""

import asyncio
import os
import sys
import re
import argparse
from pathlib import Path

import edge_tts

# Config
DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"  # Deep, clear male voice
CONTENT_DIR = Path(__file__).parent.parent / "content"
AUDIO_DIR = Path(__file__).parent.parent / "audio"

MODULES = ["neuro-boards", "ent-business", "leadership", "capital-deployment"]


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].strip()
    return text


async def generate_audio_file(text: str, output_path: Path, voice: str):
    """Generate a single MP3 file from text using edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    # Write chunks manually to avoid potential hang in save()
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])


def get_content_files(module_filter: str = None, concept_filter: str = None):
    """Get list of (module, md_path) tuples to process."""
    files = []
    for module in MODULES:
        if module_filter and module_filter.replace("_", "-") != module:
            continue
        module_dir = CONTENT_DIR / module
        if not module_dir.exists():
            continue
        for md_file in sorted(module_dir.glob("*.md")):
            if concept_filter and concept_filter.lower() not in md_file.stem.lower():
                continue
            files.append((module, md_file))
    return files


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio from content files (free, edge-tts)")
    parser.add_argument("--module", help="Module filter (e.g. neuro_boards)")
    parser.add_argument("--concept", help="Concept name filter (partial match)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Edge TTS voice name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--force", action="store_true", help="Regenerate even if audio exists")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    args = parser.parse_args()

    if args.list_voices:
        async def show_voices():
            voices = await edge_tts.list_voices()
            en_voices = [v for v in voices if v["Locale"].startswith("en-")]
            for v in sorted(en_voices, key=lambda x: x["ShortName"]):
                print(f"  {v['ShortName']:40s} {v['Gender']:8s}")
        asyncio.run(show_voices())
        return

    files = get_content_files(args.module, args.concept)
    if not files:
        print("No matching content files found.")
        sys.exit(1)

    # Calculate totals
    to_generate = []
    for module, md_path in files:
        audio_dir = AUDIO_DIR / module
        audio_path = audio_dir / md_path.with_suffix(".mp3").name
        text = strip_frontmatter(md_path.read_text(encoding="utf-8"))
        chars = len(text)
        words = len(text.split())
        est_min = round(words / 150, 1)

        if audio_path.exists() and not args.force:
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"  SKIP (exists, {size_mb:.1f}MB): {module}/{md_path.stem}")
        else:
            to_generate.append((module, md_path, text, audio_path))
            print(f"  QUEUE: {module}/{md_path.stem} ({words} words, ~{est_min} min)")

    total_words = sum(len(t.split()) for _, _, t, _ in to_generate)
    total_chars = sum(len(t) for _, _, t, _ in to_generate)

    print(f"\n{'=' * 60}")
    print(f"Total files: {len(files)}")
    print(f"To generate: {len(to_generate)}")
    print(f"Total words: {total_words:,}")
    print(f"Total chars: {total_chars:,}")
    print(f"Cost: FREE (edge-tts)")
    print(f"Voice: {args.voice}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\nDry run — no audio generated.")
        return

    if not to_generate:
        print("\nAll audio files already exist. Use --force to regenerate.")
        return

    for i, (module, md_path, text, audio_path) in enumerate(to_generate, 1):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        words = len(text.split())
        print(f"\n[{i}/{len(to_generate)}] {module}/{md_path.stem} ({words} words)...", end="", flush=True)

        try:
            asyncio.run(generate_audio_file(text, audio_path, args.voice))
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f" done ({size_mb:.1f}MB)")
        except Exception as e:
            print(f" FAILED: {e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
