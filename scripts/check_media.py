#!/usr/bin/env python3
"""Validate README media paths and browser-facing GIF/MP4 assets."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MEDIA = ROOT / "docs/media"
MAX_GITHUB_FILE_SIZE = 100 * 1024 * 1024


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def readme_media_paths(text: str) -> set[str]:
    markdown = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+\.(?:gif|png|jpe?g|mp4))\)", text, re.I)
    html = re.findall(r"(?:src|href)=[\"']([^\"']+\.(?:gif|png|jpe?g|mp4))[\"']", text, re.I)
    return {path.split("#", 1)[0] for path in [*markdown, *html]}


def check_readme(errors: list[str]) -> None:
    text = README.read_text(encoding="utf-8")
    for forbidden in ("file://", "/home/", "C:\\Users\\"):
        if forbidden in text:
            fail(errors, f"README contains a local-only path: {forbidden}")

    clickable_gif = re.search(
        r"\[!\[[^\]]*\]\(([^)]+\.gif)\)\]\(([^)]+\.mp4)\)",
        text,
        re.I,
    )
    if not clickable_gif:
        fail(errors, "README needs a clickable GIF linked to an MP4")

    for relative in sorted(readme_media_paths(text)):
        if relative.startswith(("http://", "https://")):
            continue
        if not (ROOT / relative).is_file():
            fail(errors, f"README media path does not exist: {relative}")


def check_markdown_links(errors: list[str]) -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in [README, *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve().exists():
                fail(
                    errors,
                    f"Broken local link: {path.relative_to(ROOT)} -> {target}",
                )


def check_gifs(errors: list[str]) -> None:
    gifs = sorted(MEDIA.glob("*.gif"))
    if not gifs:
        fail(errors, "No GIF found under docs/media")
    for path in gifs:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                frames = getattr(image, "n_frames", 1)
                if frames < 2:
                    fail(errors, f"GIF is not animated: {path.relative_to(ROOT)}")
                print(
                    f"GIF  {path.relative_to(ROOT)}: "
                    f"{image.width}x{image.height}, {frames} frames, {path.stat().st_size} bytes"
                )
        except Exception as exc:
            fail(errors, f"GIF decode failed for {path.relative_to(ROOT)}: {exc}")


def inspect_mp4(path: Path) -> tuple[bool, str]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height:format=duration",
                "-of",
                "default=noprint_wrappers=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0 and "codec_name=h264" in output, output.strip()

    discoverer = shutil.which("gst-discoverer-1.0")
    if discoverer:
        result = subprocess.run(
            [discoverer, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0 and "H.264" in output and "Duration:" in output, output.strip()

    data = path.read_bytes()
    valid = b"ftyp" in data[:64] and (b"avc1" in data or b"avcC" in data)
    return valid, "container signature fallback (ffprobe/gst-discoverer unavailable)"


def check_mp4s(errors: list[str]) -> None:
    videos = sorted(MEDIA.glob("*.mp4"))
    if not videos:
        fail(errors, "No MP4 found under docs/media")
    for path in videos:
        ok, details = inspect_mp4(path)
        if not ok:
            fail(errors, f"MP4 is not a readable H.264 video: {path.relative_to(ROOT)}")
        summary = next((line.strip() for line in details.splitlines() if "Duration:" in line), details)
        print(f"MP4 {path.relative_to(ROOT)}: {summary}, {path.stat().st_size} bytes")


def check_sizes(errors: list[str]) -> None:
    for path in MEDIA.glob("*"):
        if path.is_file() and path.stat().st_size >= MAX_GITHUB_FILE_SIZE:
            fail(errors, f"File exceeds GitHub's 100 MiB limit: {path.relative_to(ROOT)}")


def main() -> int:
    errors: list[str] = []
    if not README.is_file():
        fail(errors, "README.md is missing")
    if not MEDIA.is_dir():
        fail(errors, "docs/media is missing")
    if errors:
        return report(errors)

    check_readme(errors)
    check_markdown_links(errors)
    check_gifs(errors)
    check_mp4s(errors)
    check_sizes(errors)
    return report(errors)


def report(errors: list[str]) -> int:
    if errors:
        print("Media integrity check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Media integrity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
