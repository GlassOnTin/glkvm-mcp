# FastMCP server exposing the GLKVM (PiKVM kvmd fork) interactive session.
#
# Tools cover: screenshot + OCR element tree (blocks/lines/words with pixel
# bboxes), absolute mouse (with GL's 90%-width calibration), full keyboard
# (KeyboardEvent.code names, e.g. MetaLeft opens the Start menu), unicode
# typing, human-interference detection, and a raw kvmd API passthrough.

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.types import Image

from .kvmd import GlkvmClient, GlkvmError
from .vision import Engine, build_tree, flatten, find_text

mcp = FastMCP(
    "glkvm",
    instructions=(
        "Drive and observe the GLKVM KVM session (target machine screen, 2560x1440). "
        "Coordinates are target-screen pixels. Use screenshot()/screen_elements() to see the "
        "screen, then click_at/click_text/key/key_combo/type_text to act. "
        "Keyboard keys are KeyboardEvent.code names (MetaLeft/KeyA/Enter/F1/Escape/...). "
        "File transfer needs no target agent: msd_push_files wraps files in a FAT image and "
        "attaches it as a USB drive on the target; msd_pull_files streams that image back "
        "(target writes files onto the mounted drive). "
        "human_activity() warns when a human is interfering with the target machine."
    ),
)

client = GlkvmClient()
ocr = Engine()
_tree_cache: dict[str, Any] = {}  # sha -> (tree, ts)


async def _snap() -> tuple[Image, dict]:
    img, sha = await client.snapshot()
    tree = None
    if _tree_cache.get("sha") == sha:
        tree = _tree_cache.get("tree")
    info = {
        "frame_sha": sha,
        "resolution": {"width": img.width, "height": img.height},
        "ms_since_frame_capture": None,
        "cached_element_tree": tree is not None,
    }
    return img, info


def _scale_img(img, max_w: int, quality: int):
    if img.width > max_w:
        r = max_w / img.width
        img = img.resize((max_w, int(img.height * r)))
        scale = r
    else:
        scale = 1.0
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue(), scale


# ---------------------------------------------------------------- screenshot

@mcp.tool()
async def screenshot(max_width: int = 1600, quality: int = 80) -> Image:
    """Live screenshot of the target machine's screen (JPEG).

    Returns the image; screen_elements() returns the same frame's OCR tree with
    clickable bboxes in full-resolution screen pixels."""
    img, info = await _snap()
    data, _ = _scale_img(img, max_width, quality)
    return Image(data=data, format="jpeg")


# ---------------------------------------------------------------- element tree

@mcp.tool()
async def screen_elements(
    letters: bool = False,
    min_score: float = 0.45,
    max_width: int = 1600,
    force_refresh: bool = False,
) -> dict:
    """OCR the live screen into a clickable element tree.

    Structure: blocks -> lines -> words (and optionally letters), each word with
    an id, text, and bbox [x,y,w,h] in full-resolution screen pixels (2560x1440).
    Word/letter boxes inside a line are width-estimated and accurate to a few px;
    line boxes come from OCR and are precise. Also returns a downscaled image so
    you can see the frame the tree describes.

    Use force_refresh to re-OCR even if the frame hasn't changed."""
    img, sha = await client.snapshot()
    now = time.time()
    cached = _tree_cache.get("sha") == sha and not force_refresh
    if cached:
        tree = _tree_cache["tree"]
    else:
        res, ocr_ms = ocr.run(img, 1.0)
        tree = build_tree(res, img.width, img.height, sha, now, ocr_ms, letters=letters,
                          min_score=min_score)
        _tree_cache["sha"] = sha
        _tree_cache["tree"] = tree
    data, scale = _scale_img(img, max_width, 80)
    return ToolResult(
        content=[Image(data=data, format="jpeg"),
                 # JSON text block so text clients can read the tree without an image decoder
                 json.dumps(tree)],
        structured_content=tree,
    )


# ---------------------------------------------------------------- click helpers

@mcp.tool()
async def click_at(
    x: int,
    y: int,
    button: Literal["left", "right", "middle"] = "left",
    clicks: int = 1,
    gap_ms: int = 90,
) -> dict:
    """Move the target's mouse to (x, y) in screen pixels and click.

    clicks=2 for double-click. The cursor lands exactly at (x, y) up to -2/+1 px."""
    foreign = await client.interference_check()
    for i in range(max(1, clicks)):
        if i:
            await asyncio.sleep(gap_ms / 1000)
        await client.mouse_click(x, y, button)
    return {"ok": True, "clicked": [x, y], "button": button, "clicks": clicks,
            "note": "cursor tip lands at the given px",
            "interference": foreign}


@mcp.tool()
async def click_text(
    query: str,
    nth: int = 0,
    button: Literal["left", "right", "middle"] = "left",
    exact: bool = False,
    letters: bool = False,
) -> dict:
    """OCR the live screen and click the text matching query.

    Matches words and lines (case-insensitive substring unless exact=True);
    nth picks among multiple matches (default first at top-left). Returns the
    matched text and bbox. Requires a positive match to move - never guesses
    coordinates."""
    img, sha = await client.snapshot()
    tree = _tree_cache.get("tree") if (_tree_cache.get("sha") == sha) else None
    if tree is None:
        res, ocr_ms = ocr.run(img, 1.0)
        tree = build_tree(res, img.width, img.height, sha, time.time(), ocr_ms,
                          letters=letters)
        _tree_cache["sha"] = sha
        _tree_cache["tree"] = tree
    matches = find_text(tree, query, nth, exact)
    if not matches:
        return {"ok": False, "error": f"no text matching {query!r}",
                "hint": "call screen_elements() to see what is on screen", "matches": []}
    m = matches[0]
    cx, cy = m["center"]
    foreign = await client.interference_check()
    await client.mouse_click(cx, cy, button)
    return {"ok": True, "clicked": [round(cx), round(cy)], "target": m,
            "alternatives": [{"id": n["id"], "text": n["text"]} for n in matches[1:8]]}


# ---------------------------------------------------------------- mouse

@mcp.tool()
async def move_mouse(x: int, y: int) -> dict:
    """Move the target's cursor to (x, y) in screen pixels (no click)."""
    v = await client.mouse_move(x, y)
    return {"ok": True, "px": [x, y], "device_int16": v,
            "note": "reachable x range is the middle 90% of the frame (GL firmware quirk); "
                    "use mouse_relative for the extreme edges"}


@mcp.tool()
async def mouse_relative(
    dx: int = 0, dy: int = 0,
) -> dict:
    """Nudge the cursor by a relative delta (for the unreachable 5% screen edges)."""
    await client.request("POST", "/api/hid/events/send_mouse_relative",
                         query={"delta_x": max(-127, min(127, dx)),
                                "delta_y": max(-127, min(127, dy))})
    return {"ok": True, "delta": [dx, dy]}


@mcp.tool()
async def drag(
    x1: int, y1: int, x2: int, y2: int,
    button: Literal["left", "right", "middle"] = "left",
    steps: int = 14,
    settle_ms: int = 120,
) -> dict:
    """Press at (x1,y1), interpolate to (x2,y2), release. Screen-pixel coords."""
    if not (1 <= steps <= 60):
        return {"ok": False, "error": "steps must be 1..60"}
    await client.interference_check()
    await client.mouse_move(x1, y1)
    await asyncio.sleep(settle_ms / 1000)
    await client.mouse_button(button, True)
    for i in range(1, steps + 1):
        xi = x1 + (x2 - x1) * i / steps
        yi = y1 + (y2 - y1) * i / steps
        await client.mouse_move(xi, yi)
        await asyncio.sleep(0.016)
    await asyncio.sleep(settle_ms / 1000)
    await client.mouse_button(button, False)
    return {"ok": True, "drag": [[x1, y1], [x2, y2]], "button": button}


@mcp.tool()
async def scroll(
    notches: int = 3,
    direction: Literal["down", "up", "left", "right"] = "down",
    x: int | None = None,
    y: int | None = None,
) -> dict:
    """Scroll the wheel (optionally after moving to x,y). notches = wheel detents."""
    if x is not None and y is not None:
        await client.mouse_move(x, y)
    await client.scroll(notches, direction)
    return {"ok": True, "notches": notches, "direction": direction}


# ---------------------------------------------------------------- keyboard

@mcp.tool()
async def key(
    name: str,
    action: Literal["tap", "down", "up"] = "tap",
) -> dict:
    """Send a keyboard key by KeyboardEvent.code name (MetaLeft, KeyA, Enter, F5,
    Escape, Numpad1, Space, ArrowLeft, ControlLeft, ShiftLeft, AltLeft, ...).

    MetaLeft tap opens the Start menu. Use key_combo for chorded shortcuts."""
    if action == "tap":
        await client.key_tap(name)
    elif action == "down":
        await client.key_down(name)
    else:
        await client.key_up(name)
    return {"ok": True, "key": name, "action": action}


@mcp.tool()
async def key_combo(keys: list[str]) -> dict:
    """Send a chorded shortcut, e.g. ["MetaLeft", "KeyE"] (Win+E File Explorer),
    ["ControlLeft", "ShiftLeft", "KeyR"], ["AltLeft", "Tab"]. Pressed in order,
    released in reverse."""
    await client.key_combo(keys)
    return {"ok": True, "keys": keys}


@mcp.tool()
async def type_text(text: str, delay_ms: int = 10) -> dict:
    """Type unicode text on the target via kvmd's keysym printer (server-side
    layout translation). Click a text field first."""
    await client.type_text(text, delay_ms / 1000)
    return {"ok": True, "typed": text}


# ---------------------------------------------------------------- file transfer

# MSD attaches a USB drive on the target - visible to target-side security
# tooling (PnP audit, removable-storage DLP). Off unless GLKVM_ALLOW_MSD=1.
def _msd_guard() -> dict | None:
    if os.environ.get("GLKVM_ALLOW_MSD", "").strip() not in ("1", "true", "yes"):
        return {
            "ok": False,
            "disabled": True,
            "error": ("MSD file-transfer tools are disabled by default: attaching a USB "
                      "drive on the target is visible to target-side security tooling "
                      "(device-install audit, removable-storage rules). Set "
                      "GLKVM_ALLOW_MSD=1 in the glkvm server env in .mcp.json to enable."),
        }
    return None


@mcp.tool()
async def msd_state() -> dict:
    """Virtual mass-storage state: drives (cdrom / rw partition), connected,
    stored images, storage free space."""
    _g = _msd_guard()
    if _g: return _g
    return await client.msd_state()


@mcp.tool()
async def msd_push_files(
    paths: list[str],
    image_name: str = "glkvm-mcp-files.img",
    size_mb: int = 0,
    connect_target: bool = False,
    rw: bool = True,
) -> dict:
    """Push files (workstation -> target machine).

    Builds a FAT image containing the files/dirs (basenames preserved, dirs
    recursive), uploads it to the KVM's MSD storage over streaming HTTP. With
    connect_target=True the image is also attached as a USB drive on the target
    (Windows shows it immediately; disconnect later with msd_disconnect).
    Without connecting, files only reach the KVM's storage."""
    _g = _msd_guard()
    if _g: return _g
    from .msdfat import MsdFatError, build_fat_image

    os.makedirs("/tmp/glkvm-mcp-msd", exist_ok=True)
    tmp = os.path.join("/tmp/glkvm-mcp-msd", os.path.basename(image_name) or "glkvm-mcp-files.img")
    sz, files = await build_fat_image(paths, tmp, size_mb)
    written = await client.msd_write_image(tmp, os.path.basename(image_name))
    res = {"ok": True, "image": os.path.basename(image_name), "image_bytes": sz,
           "uploaded": written, "files": files, "connected": None}
    if connect_target:
        res["connected"] = await client.msd_connect(os.path.basename(image_name),
                                                    cdrom=False, rw=rw)
    return res


@mcp.tool()
async def msd_pull_files(
    image_name: str,
    out_dir: str,
    compress: Literal["lzma", "zstd", ""] = "lzma",
) -> dict:
    """Pull files (target machine -> workstation, "reversed" transfer).

    Requires the image to have been attached read-write (msd_push_files ...
    connect_target=True or msd_connect) and copied onto INSIDE the target
    (it appears as a USB drive there); then msd_disconnect() first, then this
    streams the image back (lzma by default; FAT images compress hugely),
    unpacks it with 7z into out_dir, and removes the image from KVM storage."""
    _g = _msd_guard()
    if _g: return _g
    from .msdfat import MsdFatError, unpack_fat_image

    image_name = os.path.basename(image_name)
    zpath = os.path.join(out_dir, f"{image_name}.{'xz' if compress == 'lzma' else 'zst' if compress == 'zstd' else 'raw'}")
    total = await client.msd_download_image(image_name, zpath, compress=compress)
    out_path = zpath.replace(".xz", "").replace(".zst", "").replace(".raw", "")
    if compress == "lzma":
        import lzma
        with open(zpath, "rb") as f:
            with open(out_path, "wb") as g:
                g.write(lzma.decompress(f.read()))
        os.remove(zpath)
    elif compress == "zstd":
        return {"ok": True, "downloaded": total, "downloaded_to": zpath,
                "note": "zstd: run `zstd -d` yourself or pass compress='lzma'"}
    files = await unpack_fat_image(out_path, os.path.join(out_dir, "extracted"))
    sizes = {f: os.path.getsize(f) for f in files}
    return {"ok": True, "image_bytes": os.path.getsize(out_path), "downloaded": total,
            "out_dir": os.path.join(out_dir, "extracted"), "files": files, "sizes": sizes}


@mcp.tool()
async def msd_connect(image_name: str, cdrom: bool = False, rw: bool = True) -> dict:
    """Attach a stored MSD image to the target as a USB disk (default rw flash)
    or CD-ROM (cdrom=True, always read-only). The target OS sees it immediately."""
    _g = _msd_guard()
    if _g: return _g
    return await client.msd_connect(os.path.basename(image_name), cdrom=cdrom, rw=rw)


@mcp.tool()
async def msd_disconnect() -> dict:
    """Detach the virtual drive from the target (eject first on the target if
    it is rw-mounted; kvmd busy flag guards some cases)."""
    _g = _msd_guard()
    if _g: return _g
    return await client.msd_connect(connected=False)


@mcp.tool()
async def msd_remove_image(image_name: str) -> dict:
    """Delete an image from the KVM's MSD storage."""
    _g = _msd_guard()
    if _g: return _g
    await client.msd_remove(os.path.basename(image_name))
    await asyncio.sleep(1.2)
    st = await client.msd_state()
    return {"ok": True, "images_now": list(st.get("storage", {}).get("images", {}))}


# ---------------------------------------------------------------- state / helpers

@mcp.tool()
async def human_activity(motion: bool = True) -> dict:
    """Detect concurrent human interference: compares the HID inactivity counter
    (any input into the target - ours or a human's) against this tool's last send
    time, plus an optional screen-motion probe. Returns verdict + warning flag."""
    out = await client.interference_check()
    if not motion and "screen" not in out:
        for k in ("screen",):
            out.pop(k, None)
    return out


@mcp.tool()
async def snapshot_info() -> dict:
    """Target display resolution + one fresh frame hash (cheap freshness probe)."""
    _, sha = await client.snapshot()
    return {"ok": True, "resolution": {"width": client.resolution[0], "height": client.resolution[1]},
            "frame_sha": sha}


@mcp.tool()
async def kvmd_api(method: Literal["GET", "POST"], path: str,
                   query: dict[str, Any] | None = None,
                   data: dict[str, Any] | None = None) -> Any:
    """Raw passthrough to the kvmd HTTP API (auth handled), e.g.
    GET /api/hid (LEDs), GET /api/hid/inactivity, GET /api/msd, /api/log ...

    path is the kvmd path under /api (e.g. "/api/hid"). Returns parsed JSON."""
    if not path.startswith("/api"):
        return {"error": "path must start with /api"}
    r = await client.request(method, path, query=query, data=data)
    try:
        return r.json()
    except Exception:
        return {"status": r.status_code, "text": r.text[:500]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()