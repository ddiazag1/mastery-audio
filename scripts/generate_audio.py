"""
Generate TTS audio from markdown content files using OpenAI TTS API.
Usage:
  python generate_audio.py                    # Generate all missing audio
  python generate_audio.py --module neuro_boards  # Single module
  python generate_audio.py --concept "GBM"    # Single concept (partial match)
  python generate_audio.py --dry-run          # Show what would be generated
  python generate_audio.py --model tts-1      # Use standard quality (cheaper)
"""

import os
import sys
import re
import argparse
import time
from pathlib import Path
from openai import OpenAI

# Config
VOICE = "onyx"
DEFAULT_MODEL = "tts-1-hd"  # or "tts-1" for cheaper/faster
CHUNK_LIMIT = 4096  # OpenAI TTS character limit per request
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


def chunk_text(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """Split text into chunks at sentence boundaries, respecting the char limit."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        # If a single sentence exceeds the limit, split at clause boundaries
        if len(sentence) > limit:
            if current:
                chunks.append(current.strip())
                current = ""
            # Split long sentence at commas or semicolons
            parts = re.split(r'(?<=[,;])\s+', sentence)
            for part in parts:
                if len(current) + len(part) + 1 > limit:
                    if current:
                        chunks.append(current.strip())
                    current = part
                else:
                    current = f"{current} {part}" if current else part
        elif len(current) + len(sentence) + 1 > limit:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def generate_audio_file(client: OpenAI, text: str, output_path: Path, model: str):
    """Generate a single MP3 file from text, handling chunking and concatenation."""
    chunks = chunk_text(text)
    print(f"    {len(chunks)} chunks, {len(text):,} chars")

    if len(chunks) == 1:
        # Single chunk - direct output
        response = client.audio.speech.create(
            model=model,
            voice=VOICE,
            input=chunks[0],
            response_format="mp3",
        )
        response.stream_to_file(str(output_path))
    else:
        # Multiple chunks - generate individually then concatenate
        temp_files = []
        for i, chunk in enumerate(chunks):
            temp_path = output_path.with_suffix(f".part{i:03d}.mp3")
            temp_files.append(temp_path)

            if temp_path.exists() and temp_path.stat().st_size > 0:
                print(f"    chunk {i+1}/{len(chunks)} (cached)")
                continue

            print(f"    chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...", end="", flush=True)
            retries = 3
            for attempt in range(retries):
                try:
                    response = client.audio.speech.create(
                        model=model,
                        voice=VOICE,
                        input=chunk,
                        response_format="mp3",
                    )
                    response.stream_to_file(str(temp_path))
                    print(" done")
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f" retry in {wait}s ({e})")
                        time.sleep(wait)
                    else:
                        print(f" FAILED: {e}")
                        raise

            # Rate limit courtesy
            time.sleep(0.5)

        # Concatenate MP3 chunks (binary concat works for MP3)
        with open(output_path, "wb") as outf:
            for temp_path in temp_files:
                with open(temp_path, "rb") as inf:
                    outf.write(inf.read())

        # Clean up temp files
        for temp_path in temp_files:
            temp_path.unlink(missing_ok=True)


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
    parser = argparse.ArgumentParser(description="Generate TTS audio from content files")
    parser.add_argument("--module", help="Module filter (e.g. neuro_boards)")
    parser.add_argument("--concept", help="Concept name filter (partial match)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="TTS model (tts-1 or tts-1-hd)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--force", action="store_true", help="Regenerate even if audio exists")
    args = parser.parse_args()

    files = get_content_files(args.module, args.concept)
    if not files:
        print("No matching content files found.")
        sys.exit(1)

    # Calculate totals
    total_chars = 0
    to_generate = []
    for module, md_path in files:
        audio_dir = AUDIO_DIR / module
        audio_path = audio_dir / md_path.with_suffix(".mp3").name
        text = strip_frontmatter(md_path.read_text(encoding="utf-8"))
        chars = len(text)
        total_chars += chars

        if audio_path.exists() and not args.force:
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"  SKIP (exists, {size_mb:.1f}MB): {module}/{md_path.stem}")
        else:
            to_generate.append((module, md_path, text, audio_path))
            print(f"  QUEUE: {module}/{md_path.stem} ({chars:,} chars)")

    gen_chars = sum(len(t) for _, _, t, _ in to_generate)
    cost_hd = gen_chars / 1_000_000 * 30  # $30/1M chars for tts-1-hd
    cost_std = gen_chars / 1_000_000 * 15  # $15/1M chars for tts-1

    print(f"\n{'=' * 60}")
    print(f"Total files: {len(files)}")
    print(f"To generate: {len(to_generate)}")
    print(f"Characters to generate: {gen_chars:,}")
    print(f"Est. cost (tts-1-hd): ${cost_hd:.2f}")
    print(f"Est. cost (tts-1):    ${cost_std:.2f}")
    print(f"Using model: {args.model}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\nDry run — no audio generated.")
        return

    if not to_generate:
        print("\nAll audio files already exist. Use --force to regenerate.")
        return

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    for i, (module, md_path, text, audio_path) in enumerate(to_generate, 1):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(to_generate)}] {module}/{md_path.stem}")

        try:
            generate_audio_file(client, text, audio_path, args.model)
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"  -> {audio_path} ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"  FAILED: {e}")
            # Continue with next file
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
