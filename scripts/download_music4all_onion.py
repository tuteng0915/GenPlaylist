#!/usr/bin/env python3
"""Download and verify the frozen Music4All-Onion timestamp table.

Zenodo can throttle a single transfer heavily. This downloader uses validated
HTTP byte ranges, keeps resumable parts, and publishes the file atomically only
after its size and official MD5 checksum match.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = (
    "https://zenodo.org/records/15394646/files/"
    "userid_trackid_timestamp.tsv.bz2?download=1"
)
EXPECTED_SIZE = 2_211_449_511
EXPECTED_MD5 = "dfe82201036765f7463e6f3ce3d0f991"
BUFFER_BYTES = 1024 * 1024


def _digest(path: Path, algorithm: str = "md5") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BUFFER_BYTES), b""):
            value.update(chunk)
    return value.hexdigest()


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    if total <= 0 or workers <= 0:
        raise ValueError("Download size and worker count must be positive")
    workers = min(workers, total)
    return [
        ((total * index) // workers, (total * (index + 1)) // workers - 1)
        for index in range(workers)
    ]


def _download_part(
    url: str,
    path: Path,
    start: int,
    end: int,
    retries: int,
    timeout: float,
) -> None:
    expected = end - start + 1
    for attempt in range(retries + 1):
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return
        if current > expected:
            raise ValueError(f"Oversized range part: {path}")
        first = start + current
        request = Request(
            url,
            headers={
                "Range": f"bytes={first}-{end}",
                "User-Agent": "GenPlaylist-Music4All/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise ValueError(
                        f"Server ignored byte range {first}-{end}: "
                        f"HTTP {response.status}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {first}-{end}/"):
                    raise ValueError(
                        f"Unexpected Content-Range for {path.name}: {content_range!r}"
                    )
                with path.open("ab") as handle:
                    shutil.copyfileobj(response, handle, length=BUFFER_BYTES)
            if path.stat().st_size == expected:
                return
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Range {start}-{end} failed after {retries + 1} attempts"
                ) from error
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Range {start}-{end} did not complete")


def download(
    url: str,
    output: Path,
    expected_size: int,
    expected_md5: str,
    workers: int,
    retries: int,
    timeout: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == expected_size:
        if _digest(output) == expected_md5:
            print(f"already verified: {output}")
            return
        raise ValueError(f"Existing full-size file has the wrong checksum: {output}")

    ranges = _ranges(expected_size, workers)
    parts_dir = output.with_name(output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    first_part = parts_dir / "part-000.bin"
    if output.exists():
        partial_size = output.stat().st_size
        first_size = ranges[0][1] - ranges[0][0] + 1
        if partial_size <= first_size and not first_part.exists():
            os.replace(output, first_part)
            print(f"reused {partial_size} bytes from the earlier single-stream download")
        else:
            raise ValueError(
                f"Cannot safely reuse incomplete output {output}; "
                f"move it aside and retry"
            )

    futures = {}
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        for index, (start, end) in enumerate(ranges):
            part = parts_dir / f"part-{index:03d}.bin"
            future = executor.submit(
                _download_part, url, part, start, end, retries, timeout)
            futures[future] = (index, start, end)
        for future in as_completed(futures):
            index, start, end = futures[future]
            future.result()
            print(f"completed part {index + 1}/{len(ranges)} ({start}-{end})", flush=True)

    temporary = output.with_name(output.name + ".assembling")
    with temporary.open("wb") as destination:
        for index, _ in enumerate(ranges):
            with (parts_dir / f"part-{index:03d}.bin").open("rb") as source:
                shutil.copyfileobj(source, destination, length=BUFFER_BYTES)
        destination.flush()
        os.fsync(destination.fileno())
    if temporary.stat().st_size != expected_size:
        raise ValueError("Assembled Music4All-Onion file has the wrong size")
    actual_md5 = _digest(temporary)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Music4All-Onion checksum mismatch: {actual_md5} != {expected_md5}")
    os.replace(temporary, output)
    for index, _ in enumerate(ranges):
        (parts_dir / f"part-{index:03d}.bin").unlink()
    parts_dir.rmdir()
    print(f"verified {output} (md5={actual_md5})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--expected-size", type=int, default=EXPECTED_SIZE)
    parser.add_argument("--expected-md5", default=EXPECTED_MD5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.retries < 0 or args.timeout <= 0:
        raise ValueError("Retries must be nonnegative and timeout positive")
    download(
        args.url,
        args.output.expanduser().resolve(),
        args.expected_size,
        args.expected_md5.casefold(),
        args.workers,
        args.retries,
        args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
