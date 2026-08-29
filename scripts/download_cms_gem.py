"""Download immutable authoritative CMS resources without preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cms_resources import RESOURCES, raw_directory

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resource(resource, destination: Path) -> str:
    if destination.exists():
        digest = sha256_file(destination)
        if digest != resource.sha256:
            raise RuntimeError(
                f"Refusing to overwrite {destination}: existing SHA-256 {digest} "
                f"does not match expected {resource.sha256}."
            )
        print(
            f"already present: {destination} ({destination.stat().st_size} bytes; SHA-256 matches)"
        )
        return digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(resource.url, headers={"User-Agent": "SHIFT-ICD/0.1 data-provenance"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {resource.url}: {exc}") from exc

    digest = sha256_file(temporary)
    if digest != resource.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {resource.url}: downloaded {digest}, "
            f"expected {resource.sha256}."
        )
    temporary.replace(destination)
    print(f"downloaded: {destination} ({destination.stat().st_size} bytes; SHA-256 matches)")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = raw_directory(root)
    for resource in RESOURCES:
        download_resource(resource, output_dir / resource.filename)
    return 0


if __name__ == "__main__":
    sys.exit(main())
