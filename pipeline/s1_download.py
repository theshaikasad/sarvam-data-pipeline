"""Stage 1: download + normalize source audio.

For each channel in config.yaml, download audio with yt-dlp and produce a clean,
loudness-normalized 16kHz mono WAV at data/raw/<channel>_<videoid>.wav.

s1 does NOT create manifest rows — s2 scans data/raw/ and creates one row per clip.
The raw filename (<channel>_<videoid>.wav) carries everything s2 needs: the channel
name (to look up language/source_type in config) and the YouTube id (for source_url).

Resumable:
  - yt-dlp uses a download archive (data/raw/.yt-archive.txt) to skip videos already
    fetched.
  - ffmpeg normalization is skipped if the final <channel>_<videoid>.wav already exists.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import yaml

CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_channel(channel: dict, raw_dir: str, archive: str) -> None:
    """Fetch audio for one channel's url_or_playlist into <channel>_<id>.src.<ext>."""
    name = channel["name"]
    url = channel.get("url_or_playlist")
    if not url or "XXXX" in str(url) or not str(url).startswith("http"):
        print(f"  [skip] {name}: no real URL set (placeholder).")
        return
    max_videos = int(channel.get("max_videos", 1))
    out_tmpl = os.path.join(raw_dir, f"{name}_%(id)s.src.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--download-archive", archive,
        "--playlist-end", str(max_videos),
        "--no-overwrites",
        "-o", out_tmpl,
        url,
    ]
    print(f"  [download] {name}: {url} (<= {max_videos} videos)")
    subprocess.run(cmd, check=True)


def normalize(src: str, dst: str, sample_rate: int, target_lufs: float) -> None:
    """ffmpeg: loudness-normalize toward target_lufs and resample to 16kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-af", f"loudnorm=I={target_lufs}:TP=-2:LRA=11",
        "-ar", str(sample_rate),
        "-ac", "1",
        dst,
    ]
    subprocess.run(cmd, check=True)


def run(config_path: str = CONFIG_PATH) -> None:
    cfg = load_config(config_path)
    raw_dir = cfg["paths"]["raw"]
    sample_rate = int(cfg["sample_rate"])
    target_lufs = float(cfg["target_lufs"])
    os.makedirs(raw_dir, exist_ok=True)
    archive = os.path.join(raw_dir, ".yt-archive.txt")

    # 1) download
    for channel in cfg["channels"]:
        try:
            download_channel(channel, raw_dir, archive)
        except subprocess.CalledProcessError as e:
            print(f"  [error] download failed for {channel['name']}: {e}")

    # 2) normalize every .src.wav -> final <channel>_<id>.wav
    for src in sorted(glob.glob(os.path.join(raw_dir, "*.src.wav"))):
        dst = src.replace(".src.wav", ".wav")
        if os.path.exists(dst):
            print(f"  [skip] already normalized: {os.path.basename(dst)}")
            continue
        print(f"  [normalize] {os.path.basename(src)} -> {os.path.basename(dst)}")
        try:
            normalize(src, dst, sample_rate, target_lufs)
            os.remove(src)
        except subprocess.CalledProcessError as e:
            print(f"  [error] normalize failed for {src}: {e}")

    print("s1 done.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
