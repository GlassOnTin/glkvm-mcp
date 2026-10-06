# FAT-image wrapping for MSD file transfer.
#
# kvmd MSD emits whole disk images, not files. To move files we build a small
# FAT filesystem image locally (mkfs.vfat + mcopy), upload it, and unpack it on
# the other side (7z). The sync subprocess work runs off the asyncio loop.

from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess


class MsdFatError(RuntimeError):
    pass


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise MsdFatError(f"{cmd[0]} failed ({p.returncode}): {p.stderr.strip()[:300]}")
    return p.stdout


def _tree_size(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p) + 2048  # + per-file directory entry slack
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total + 4096  # root directory slack


async def build_fat_image(paths: list[str], out_path: str,
                          size_mb: int = 0) -> tuple[int, list[str]]:
    """Create a FAT image at out_path containing the given files/dirs (top-level
    entries keep their basenames; dirs copy recursively via mcopy -s).

    Auto-sized to payload*1.4 + 8 MB unless size_mb is given.
    Returns (image_size_bytes, resolved absolute paths)."""
    if not paths:
        raise MsdFatError("no input paths given")
    specs: list[tuple[str, str | None]] = []
    names: set[str] = set()
    payload = 0
    for raw in paths:
        p = os.path.abspath(os.path.expanduser(raw))
        base = os.path.basename(p.rstrip("/"))
        if not os.path.exists(p):
            raise MsdFatError(f"no such path: {p}")
        if base in names:
            raise MsdFatError(f"duplicate basename {base!r}; rename first")
        names.add(base)
        specs.append((p, None if os.path.isdir(p) else base))
        payload += _tree_size(p)
    if size_mb <= 0:
        size_mb = max(8, math.ceil(payload * 1.4 / 1048576) + 8)
    size_bytes = size_mb * 1048576
    if payload > size_bytes - 1048576:
        raise MsdFatError(f"payload {payload} B does not fit in {size_mb} MB; pass size_mb")
    await asyncio.to_thread(_build, specs, out_path, size_bytes)
    return size_bytes, [s[0] for s in specs]


def _build(specs: list[tuple[str, str | None]], out_path: str, size_bytes: int) -> None:
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path, "wb") as f:
        f.truncate(size_bytes)
    _run(["mkfs.vfat", "-n", "GLKVMMCP", out_path])
    for (src, name) in specs:
        arg = f"::{name}" if name else "::"
        _run(["mcopy", "-i", out_path, "-s", src, arg])


async def unpack_fat_image(img_path: str, out_dir: str) -> list[str]:
    """Extract the files of a FAT image into out_dir (uses 7z). Returns the
    extracted file paths (empty dirs skipped)."""
    if shutil.which("7z") is None:
        raise MsdFatError("7z (p7zip-full) not installed on the workstation")
    os.makedirs(out_dir, exist_ok=True)
    await asyncio.to_thread(_run, ["7z", "x", "-y", f"-o{out_dir}", img_path])
    out = []
    for root, _, files in os.walk(out_dir):
        out.extend(os.path.join(root, f) for f in files)
    return sorted(out)