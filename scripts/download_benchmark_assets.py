#!/usr/bin/env python3
"""Download benchmark assets (JSUT and LibriSpeech) if not present.

This script downloads the required speech corpora for ASR benchmarking:
- JSUT: Japanese speech corpus (basic5000 subset for standard mode)
- LibriSpeech test-clean: English speech corpus

The corpora are not included in git due to licensing restrictions.

Usage:
    python scripts/download_benchmark_assets.py

    # Download only specific corpus
    python scripts/download_benchmark_assets.py --ja-only
    python scripts/download_benchmark_assets.py --en-only

    # Force re-download
    python scripts/download_benchmark_assets.py --force

Environment Variables:
    LIVECAP_JSUT_DIR: Custom path for JSUT (default: tests/assets/source/jsut/jsut_ver1.1)
    LIVECAP_LIBRISPEECH_DIR: Custom path for LibriSpeech (default: tests/assets/source/librispeech/test-clean)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"
SOURCE_DIR = ASSETS_DIR / "source"

# Default paths
DEFAULT_JSUT_DIR = SOURCE_DIR / "jsut" / "jsut_ver1.1"
DEFAULT_LIBRISPEECH_DIR = SOURCE_DIR / "librispeech" / "test-clean"

# Download URLs
# LibriSpeech test-clean from OpenSLR (official)
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"

# JSUT from official source (Takamichi Lab server)
# Direct download link from the corpus author's server
JSUT_URL = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
JSUT_MD5 = None  # MD5 not provided by source

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def calculate_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_with_progress(url: str, dest: Path) -> bool:
    """Download a file with progress reporting.

    Args:
        url: URL to download
        dest: Destination path

    Returns:
        True if successful, False otherwise
    """
    def report_progress(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Downloading: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent}%)", end="", flush=True)

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, dest, reporthook=report_progress)
        print()  # New line after progress
        return True
    except URLError as e:
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return False


def extract_tar_gz(archive: Path, dest_dir: Path) -> bool:
    """Extract a tar.gz archive.

    Args:
        archive: Path to archive
        dest_dir: Destination directory

    Returns:
        True if successful
    """
    try:
        logger.info(f"  Extracting to {dest_dir}...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest_dir)
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def extract_zip(archive: Path, dest_dir: Path) -> bool:
    """Extract a zip archive.

    Args:
        archive: Path to archive
        dest_dir: Destination directory

    Returns:
        True if successful
    """
    try:
        logger.info(f"  Extracting to {dest_dir}...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_librispeech(dest_dir: Path, force: bool = False) -> bool:
    """Download LibriSpeech test-clean corpus.

    Args:
        dest_dir: Destination directory for test-clean/
        force: Force re-download even if exists

    Returns:
        True if successful or already exists
    """
    # Check if already exists
    # LibriSpeech extracts to: LibriSpeech/test-clean/
    test_clean_dir = dest_dir
    if test_clean_dir.exists() and not force:
        # Verify it has content
        speaker_dirs = list(test_clean_dir.glob("*/"))
        if speaker_dirs:
            logger.info(f"LibriSpeech test-clean already exists at {test_clean_dir}")
            return True

    logger.info("=== Downloading LibriSpeech test-clean ===")
    logger.info(f"  URL: {LIBRISPEECH_URL}")
    logger.info(f"  Size: ~350 MB")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "test-clean.tar.gz"

        # Download
        if not download_with_progress(LIBRISPEECH_URL, archive_path):
            return False

        # Verify MD5 if available
        if LIBRISPEECH_MD5:
            logger.info("  Verifying checksum...")
            actual_md5 = calculate_md5(archive_path)
            if actual_md5 != LIBRISPEECH_MD5:
                logger.error(f"MD5 mismatch: expected {LIBRISPEECH_MD5}, got {actual_md5}")
                return False
            logger.info("  Checksum OK")

        # Extract
        extract_dir = Path(tmpdir) / "extract"
        if not extract_tar_gz(archive_path, extract_dir):
            return False

        # Move to destination
        # LibriSpeech extracts to: extract/LibriSpeech/test-clean/
        extracted = extract_dir / "LibriSpeech" / "test-clean"
        if not extracted.exists():
            logger.error(f"Expected directory not found: {extracted}")
            return False

        # Create parent and move
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(extracted), str(dest_dir))

        logger.info(f"  Installed to: {dest_dir}")

    return True


def download_jsut(dest_dir: Path, force: bool = False) -> bool:
    """Download JSUT corpus.

    Args:
        dest_dir: Destination directory for jsut_ver1.1/
        force: Force re-download even if exists

    Returns:
        True if successful or already exists
    """
    # Check if already exists
    if dest_dir.exists() and not force:
        # Verify it has content (basic5000 subset)
        basic5000 = dest_dir / "basic5000"
        if basic5000.exists():
            logger.info(f"JSUT already exists at {dest_dir}")
            return True

    logger.info("=== Downloading JSUT ===")
    logger.info(f"  URL: {JSUT_URL}")
    logger.info(f"  Size: ~2 GB (this may take a while)")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "jsut_ver1.1.zip"

        # Download
        if not download_with_progress(JSUT_URL, archive_path):
            return False

        # Verify MD5 if available
        if JSUT_MD5:
            logger.info("  Verifying checksum...")
            actual_md5 = calculate_md5(archive_path)
            if actual_md5 != JSUT_MD5:
                logger.warning(f"MD5 mismatch: expected {JSUT_MD5}, got {actual_md5}")
                # Don't fail on MD5 mismatch for JSUT since Zenodo MD5 varies

        # Extract
        extract_dir = Path(tmpdir) / "extract"
        if not extract_zip(archive_path, extract_dir):
            return False

        # Find jsut_ver1.1 directory
        # It should be directly in extract/ or nested
        extracted = extract_dir / "jsut_ver1.1"
        if not extracted.exists():
            # Try to find it
            candidates = list(extract_dir.glob("**/jsut_ver1.1"))
            if candidates:
                extracted = candidates[0]
            else:
                # Maybe the zip extracts contents directly
                candidates = list(extract_dir.glob("**/basic5000"))
                if candidates:
                    extracted = candidates[0].parent
                else:
                    logger.error(f"JSUT structure not found in archive")
                    return False

        # Create parent and move
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(extracted), str(dest_dir))

        logger.info(f"  Installed to: {dest_dir}")

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download benchmark assets (JSUT and LibriSpeech)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if assets exist",
    )
    parser.add_argument(
        "--ja-only",
        action="store_true",
        help="Download only JSUT (Japanese)",
    )
    parser.add_argument(
        "--en-only",
        action="store_true",
        help="Download only LibriSpeech (English)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get paths from environment or defaults
    jsut_dir = Path(os.getenv("LIVECAP_JSUT_DIR", str(DEFAULT_JSUT_DIR)))
    librispeech_dir = Path(os.getenv("LIVECAP_LIBRISPEECH_DIR", str(DEFAULT_LIBRISPEECH_DIR)))

    logger.info("Benchmark Asset Downloader")
    logger.info(f"JSUT destination: {jsut_dir}")
    logger.info(f"LibriSpeech destination: {librispeech_dir}")
    logger.info("")

    success = True

    # Download JSUT
    if not args.en_only:
        if not download_jsut(jsut_dir, args.force):
            logger.error("Failed to download JSUT")
            success = False

    # Download LibriSpeech
    if not args.ja_only:
        if not download_librispeech(librispeech_dir, args.force):
            logger.error("Failed to download LibriSpeech")
            success = False

    if success:
        logger.info("")
        logger.info("=== Download Complete ===")
        logger.info("Run benchmark data preparation with:")
        logger.info("  python scripts/prepare_benchmark_data.py --mode standard")
    else:
        logger.error("")
        logger.error("=== Download Failed ===")
        logger.error("Please check the error messages above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
