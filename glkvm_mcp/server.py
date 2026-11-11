# FastMCP server exposing the GLKVM (PiKVM kvmd fork) interactive session.
#
# Tools cover: screenshot + OCR element tree/lines/diff/text, absolute mouse
# (with automatic pillarbox detection), full keyboard (KeyboardEvent.code
# names, e.g. MetaLeft opens the Start menu), layout-aware typing, multi-step
# scrollable pane reading, human-interference detection, EDID re-emit (HPD
# pulse to wake a target whose GPU dropped its HDMI output), and a raw kvmd
# API passthrough.

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from typing import Any, Literal

import httpx
from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.types import Image

from .kvmd import GlkvmClient, GlkvmError, LAYOUT
from .vision import (
    Engine,
    anchor_point,
    build_tree,
    diff_lines,
    extract_lines,
    extract_text_rows,
    find_text,
    flatten,
    _norm,
)

mcp = FastMCP(
    "glkvm",
    instructions=(
        "Drive and observe the GLKVM KVM session. "
        "Coordinates are full target-screen pixels (returned in screenshot/screen_elements metadata; "
        "1080p screenshots are 1:1 by default at max_width=1920). "
        "Prefer screen_elements(mode='lines' or 'text' or 'diff', include_image=False) for compact, "
        "low-token OCR, or pass region=[x,y,w,h] to restrict OCR while keeping full-screen bboxes. "
        "Pass inspect_after='diff'|'lines'|'text' or return_screenshot=True on click_at, click_text, "
        "type_text, key, key_combo, or scroll to act and observe in a single round-trip. "
        "Use read_scrollable() to read multi-page terminal/TUI/browser panes via wheel-scroll + OCR "
        "without sending any keystrokes. "
        "Keyboard keys are KeyboardEvent.code names (MetaLeft/KeyA/Enter/F1/Escape/...). "
        "If HDMI signal dropped after a KVM reboot, _snap() re-emits EDID once automatically; "
        "edid_prod() triggers that nudge on demand and video_state() checks signal state. "
        "human_activity() checks for concurrent human input."
    ),
)

client = GlkvmClient()
ocr = Engine()
_tree_cache: dict[tuple, dict] = {}
_last_lines_snapshot: dict[tuple, list[dict]] = {}


def _norm_region(region: list[int] | None, w: int, h: int) -> tuple[int, int, int, int] | None:
    if not region:
        return None
    if len(region) != 4:
        raise GlkvmError("region must be [x, y, width, height]")
    rx = max(0, min(w - 1, int(region[0])))
    ry = max(0, min(h - 1, int(region[1])))
    rw = max(4, min(w - rx, int(region[2])))
    rh = max(4, min(h - ry, int(region[3])))
    return (rx, ry, rw, rh)


async def _snap() -> tuple[Any, dict]:
    prod = None
    try:
        img, sha = await client.snapshot()
    except (GlkvmError, httpx.HTTPError, OSError) as e:
        # no frame: if the target's GPU dropped its HDMI output (usual after a
        # KVM reboot), re-emitting EDID pulses HPD and wakes it (GL UI flick).
        # OSError covers PIL's UnidentifiedImageError if the endpoint returns
        # a non-image body instead of an error status while there is no signal.
        try:
            st = await client.video_state()
        except Exception:
            raise e
        if st.get("signal") is not False:
            raise GlkvmError(f"snapshot failed while HDMI signal is {st.get('signal')!r}: {e}") from e
        p = await client.edid_prod(timeout_s=15.0)
        if not p["ok"]:
            raise GlkvmError(
                "screenshot failed: no HDMI signal, and re-emitting EDID did not "
                f"recover it (raw: {e})") from e
        img, sha = await client.snapshot()
        prod = {"note": "no HDMI signal; re-emitted EDID (HPD pulse) and recovered",
                "state": p}
    active_l, active_w = client.active_bounds
    info: dict[str, Any] = {
        "frame_sha": sha,
        "width": img.width,
        "height": img.height,
        "resolution": {"width": img.width, "height": img.height},
        "active_left_px": active_l,
        "active_width_px": active_w,
    }
    if prod:
        info["edid_prod"] = prod
    return img, info


def _scale_img(img, max_w: int, quality: int) -> tuple[bytes, float, int, int]:
    if max_w > 0 and img.width > max_w:
        r = max_w / img.width
        new_h = max(1, round(img.height * r))
        img = img.resize((max_w, new_h))
        scale = round(r, 6)
    else:
        scale = 1.0
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue(), scale, img.width, img.height


def _run_or_get_tree(
    img,
    sha: str,
    *,
    region: list[int] | None = None,
    letters: bool = False,
    min_score: float = 0.45,
    force_refresh: bool = False,
) -> tuple[dict, tuple]:
    reg = _norm_region(region, img.width, img.height)
    reg_key = reg or (0, 0, img.width, img.height)
    cache_key = (sha, reg_key, bool(letters), round(min_score, 3))
    if not force_refresh and cache_key in _tree_cache:
        return _tree_cache[cache_key], reg_key
    crop = img.crop((reg[0], reg[1], reg[0] + reg[2], reg[1] + reg[3])) if reg else img
    ox, oy = (reg[0], reg[1]) if reg else (0, 0)
    res, ocr_ms = ocr.run(crop, 1.0)
    tree = build_tree(
        res,
        img.width,
        img.height,
        sha,
        time.time(),
        ocr_ms,
        letters=letters,
        min_score=min_score,
        offset_x=ox,
        offset_y=oy,
        region=list(reg) if reg else None,
    )
    if len(_tree_cache) >= 8:
        _tree_cache.clear()
    _tree_cache[cache_key] = tree
    return tree, reg_key


async def _capture_pre_baseline(
    inspect_after: str,
    region: list[int] | None = None,
) -> None:
    if inspect_after != "diff":
        return
    try:
        img, info = await _snap()
        tree, reg_key = _run_or_get_tree(img, info["frame_sha"], region=region)
        _last_lines_snapshot[reg_key] = extract_lines(tree)
    except Exception:
        pass


async def _with_post_action(
    payload: dict[str, Any],
    *,
    inspect_after: Literal["none", "lines", "diff", "text"] = "none",
    return_screenshot: bool = False,
    settle_ms: int = 250,
    region: list[int] | None = None,
    max_width: int = 1920,
    quality: int = 80,
) -> Any:
    want_inspect = inspect_after != "none"
    if not want_inspect and not return_screenshot:
        return payload
    wait_s = max(0, min(5000, settle_ms)) / 1000.0
    if wait_s > 0:
        await asyncio.sleep(wait_s)
    img, info = await _snap()
    if want_inspect:
        tree, reg_key = _run_or_get_tree(
            img, info["frame_sha"], region=region, force_refresh=True
        )
        curr_lines = extract_lines(tree)
        prev_lines = _last_lines_snapshot.get(reg_key)
        _last_lines_snapshot[reg_key] = curr_lines
        if inspect_after == "diff":
            payload["inspection"] = {
                "frame": tree["frame"],
                **diff_lines(prev_lines, curr_lines),
            }
        elif inspect_after == "text":
            payload["inspection"] = {
                "frame": tree["frame"],
                "lines": extract_text_rows(tree),
            }
        else:
            payload["inspection"] = {
                "frame": tree["frame"],
                "lines": curr_lines,
            }
    if not return_screenshot:
        return payload
    reg = _norm_region(region, img.width, img.height)
    view = img.crop((reg[0], reg[1], reg[0] + reg[2], reg[1] + reg[3])) if reg else img
    data, scale, iw, ih = _scale_img(view, max_width, quality)
    payload["screenshot"] = {
        **info,
        "image_width": iw,
        "image_height": ih,
        "scale": scale,
        **({"region": list(reg)} if reg else {}),
    }
    return ToolResult(
        content=[Image(data=data, format="jpeg"), json.dumps(payload)],
        structured_content=payload,
    )


# ---------------------------------------------------------------- screenshot

@mcp.tool()
async def screenshot(
    max_width: int = 1920,
    quality: int = 80,
    region: list[int] | None = None,
) -> Any:
    """Live screenshot of the target machine's screen (JPEG) plus structured resolution/scale metadata.

    Defaults to max_width=1920 so 1080p frames are 1:1 with click coordinates (scale=1.0).
    Pass region=[x, y, width, height] in full-screen pixels to crop a region of interest."""
    img, info = await _snap()
    reg = _norm_region(region, img.width, img.height)
    view = img.crop((reg[0], reg[1], reg[0] + reg[2], reg[1] + reg[3])) if reg else img
    data, scale, iw, ih = _scale_img(view, max_width, quality)
    meta = {
        **info,
        "image_width": iw,
        "image_height": ih,
        "scale": scale,
    }
    if reg:
        meta["region"] = list(reg)
    return ToolResult(
        content=[Image(data=data, format="jpeg"), json.dumps(meta)],
        structured_content=meta,
    )


# ---------------------------------------------------------------- element tree

@mcp.tool()
async def screen_elements(
    mode: Literal["tree", "lines", "text", "find", "diff"] = "tree",
    query: str | None = None,
    exact: bool = False,
    near: list[int] | None = None,
    region: list[int] | None = None,
    include_image: bool = True,
    letters: bool = False,
    min_score: float = 0.45,
    max_width: int = 1920,
    force_refresh: bool = False,
) -> Any:
    """OCR the live screen in full-screen target pixel coordinates.

    Modes:
      - 'lines': compact [{id, text, score, bbox, center}] without per-word bloat (~6x smaller)
      - 'text': row-merged plain text lines (fastest/smallest way to read terminal or code text)
      - 'find': search for `query` (with optional `exact=True` and `near=[x,y]` proximity ranking)
      - 'diff': added_lines and removed_lines compared to the previous inspection on this region
      - 'tree' (default): full blocks -> lines -> words hierarchy

    Set include_image=False to omit the JPEG when you only want OCR text/coordinates.
    Pass region=[x, y, width, height] to OCR a sub-region while keeping bboxes in full-screen pixels."""
    img, info = await _snap()
    tree, reg_key = _run_or_get_tree(
        img,
        info["frame_sha"],
        region=region,
        letters=letters,
        min_score=min_score,
        force_refresh=force_refresh,
    )
    curr_lines = extract_lines(tree)
    prev_lines = _last_lines_snapshot.get(reg_key)
    _last_lines_snapshot[reg_key] = curr_lines

    resolved_mode = "find" if (mode == "tree" and query) else mode
    active_l, active_w = client.active_bounds
    base_meta = {
        **tree["frame"],
        "active_left_px": active_l,
        "active_width_px": active_w,
    }
    if resolved_mode == "lines":
        out: dict[str, Any] = {"frame": base_meta, "mode": "lines", "lines": curr_lines}
    elif resolved_mode == "text":
        rows = extract_text_rows(tree)
        out = {"frame": base_meta, "mode": "text", "line_count": len(rows), "lines": rows}
    elif resolved_mode == "find":
        matches = find_text(tree, query or "", nth=0, exact=exact, near=near)
        out = {
            "frame": base_meta,
            "mode": "find",
            "query": query or "",
            "match_count": len(matches),
            "matches": matches[:25],
        }
    elif resolved_mode == "diff":
        out = {
            "frame": base_meta,
            "mode": "diff",
            **diff_lines(prev_lines, curr_lines),
        }
    else:
        out = {**tree, "frame": base_meta}

    content_blocks: list[Any] = []
    if include_image:
        reg = _norm_region(region, img.width, img.height)
        view = img.crop((reg[0], reg[1], reg[0] + reg[2], reg[1] + reg[3])) if reg else img
        data, scale, iw, ih = _scale_img(view, max_width, 80)
        out["frame"]["image_w"] = iw
        out["frame"]["image_h"] = ih
        out["frame"]["scale"] = scale
        content_blocks.append(Image(data=data, format="jpeg"))
    content_blocks.append(json.dumps(out))
    return ToolResult(content=content_blocks, structured_content=out)


# ---------------------------------------------------------------- click helpers

@mcp.tool()
async def click_at(
    x: int = 0,
    y: int = 0,
    button: Literal["left", "right", "middle"] = "left",
    clicks: int = 1,
    gap_ms: int = 90,
    points: list[dict[str, Any]] | None = None,
    check_interference: bool = False,
    inspect_after: Literal["none", "lines", "diff", "text"] = "none",
    return_screenshot: bool = False,
    settle_ms: int = 250,
    region: list[int] | None = None,
) -> Any:
    """Move the target's mouse to (x, y) in full-screen pixels and click (or click `points` in sequence).

    Pass points=[{"x": 100, "y": 200, "button": "left", "clicks": 1, "delay_ms": 80}, ...] to batch
    multiple clicks in one round-trip. Set inspect_after='diff'|'lines'|'text' or return_screenshot=True
    to observe the post-click screen in the same call."""
    await _capture_pre_baseline(inspect_after, region)
    foreign = await client.interference_check() if check_interference else None
    if points:
        clicked_pts = []
        for idx, pt in enumerate(points[:20]):
            px, py = int(pt["x"]), int(pt["y"])
            btn = str(pt.get("button", button))
            clk = max(1, min(3, int(pt.get("clicks", clicks))))
            d_ms = max(0, min(5000, int(pt.get("delay_ms", 0 if idx == 0 else 80))))
            if d_ms > 0:
                await asyncio.sleep(d_ms / 1000)
            for i in range(clk):
                if i:
                    await asyncio.sleep(gap_ms / 1000)
                await client.mouse_click(px, py, btn)
            clicked_pts.append({"x": px, "y": py, "button": btn, "clicks": clk})
        res: dict[str, Any] = {
            "ok": True,
            "clicked": [clicked_pts[-1]["x"], clicked_pts[-1]["y"]],
            "points_clicked": clicked_pts,
        }
    else:
        for i in range(max(1, clicks)):
            if i:
                await asyncio.sleep(gap_ms / 1000)
            await client.mouse_click(x, y, button)
        res = {"ok": True, "clicked": [x, y], "button": button, "clicks": clicks}
    if foreign is not None:
        res["interference"] = foreign
    return await _with_post_action(
        res,
        inspect_after=inspect_after,
        return_screenshot=return_screenshot,
        settle_ms=settle_ms,
        region=region,
    )


@mcp.tool()
async def click_text(
    query: str,
    nth: int = 0,
    button: Literal["left", "right", "middle"] = "left",
    clicks: int = 1,
    exact: bool = False,
    near: list[int] | None = None,
    anchor: Literal["center", "left", "right", "top", "bottom"] = "center",
    dx: int = 0,
    dy: int = 0,
    region: list[int] | None = None,
    letters: bool = False,
    check_interference: bool = False,
    inspect_after: Literal["none", "lines", "diff", "text"] = "none",
    return_screenshot: bool = False,
    settle_ms: int = 250,
) -> Any:
    """OCR the live screen and click the text matching query.

    Supports `near=[x,y]` proximity disambiguation, `anchor` ('center'|'left'|'right'|'top'|'bottom')
    plus `(dx, dy)` pixel offset (e.g. anchor='right', dx=12 for a dropdown chevron), `region` ROI,
    and post-click `inspect_after` / `return_screenshot`."""
    img, info = await _snap()
    tree, reg_key = _run_or_get_tree(img, info["frame_sha"], region=region, letters=letters)
    if inspect_after == "diff":
        _last_lines_snapshot[reg_key] = extract_lines(tree)
    matches = find_text(tree, query, nth, exact, near=near)
    if not matches:
        return {
            "ok": False,
            "error": f"no text matching {query!r}",
            "hint": "call screen_elements(mode='lines') to see what is on screen",
            "matches": [],
        }
    m = matches[0]
    cx, cy = anchor_point(m["bbox"], anchor=anchor, dx=dx, dy=dy, frame_w=img.width, frame_h=img.height)
    foreign = await client.interference_check() if check_interference else None
    for i in range(max(1, min(3, clicks))):
        if i:
            await asyncio.sleep(0.06)
        await client.mouse_click(cx, cy, button)
    res: dict[str, Any] = {
        "ok": True,
        "clicked": [cx, cy],
        "anchor": anchor,
        "target": m,
        "alternatives": [{"id": n["id"], "text": n["text"], "bbox": n["bbox"]} for n in matches[1:8]],
    }
    if foreign is not None:
        res["interference"] = foreign
    return await _with_post_action(
        res,
        inspect_after=inspect_after,
        return_screenshot=return_screenshot,
        settle_ms=settle_ms,
        region=region,
    )


# ---------------------------------------------------------------- mouse

@mcp.tool()
async def move_mouse(x: int, y: int) -> dict:
    """Move the target's cursor to (x, y) in full-screen pixels (no click)."""
    v = await client.mouse_move(x, y)
    active_l, active_w = client.active_bounds
    return {
        "ok": True,
        "px": [x, y],
        "device_int16": v,
        "active_bounds": {"left_px": active_l, "width_px": active_w},
    }


@mcp.tool()
async def mouse_relative(
    dx: int = 0, dy: int = 0,
) -> dict:
    """Nudge the cursor by a relative delta (-127..127 per call)."""
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
    check_interference: bool = False,
) -> dict:
    """Press at (x1,y1), interpolate to (x2,y2), release. Screen-pixel coords."""
    if not (1 <= steps <= 60):
        return {"ok": False, "error": "steps must be 1..60"}
    if check_interference:
        await client.interference_check()
    await client.mouse_move(x1, y1)
    await asyncio.sleep(settle_ms / 1000)
    await client.mouse_button(button, True)
    try:
        for i in range(1, steps + 1):
            xi = x1 + (x2 - x1) * i / steps
            yi = y1 + (y2 - y1) * i / steps
            await client.mouse_move(xi, yi)
            await asyncio.sleep(0.016)
        await asyncio.sleep(settle_ms / 1000)
    finally:
        await client.mouse_button(button, False)
    return {"ok": True, "drag": [[x1, y1], [x2, y2]], "button": button}


@mcp.tool()
async def scroll(
    notches: int = 3,
    direction: Literal["down", "up", "left", "right"] = "down",
    x: int | None = None,
    y: int | None = None,
    inspect_after: Literal["none", "lines", "diff", "text"] = "none",
    return_screenshot: bool = False,
    settle_ms: int = 250,
    region: list[int] | None = None,
) -> Any:
    """Scroll the wheel (optionally after moving to x,y). notches = wheel detents."""
    await _capture_pre_baseline(inspect_after, region)
    if x is not None and y is not None:
        await client.mouse_move(x, y)
    await client.scroll(notches, direction)
    res: dict[str, Any] = {"ok": True, "notches": notches, "direction": direction}
    if x is not None and y is not None:
        res["px"] = [x, y]
    return await _with_post_action(
        res,
        inspect_after=inspect_after,
        return_screenshot=return_screenshot,
        settle_ms=settle_ms,
        region=region,
    )


@mcp.tool()
async def read_scrollable(
    x: int,
    y: int,
    direction: Literal["up", "down"] = "up",
    steps: int = 10,
    notches_per_step: int = 8,
    region: list[int] | None = None,
    settle_ms: int = 250,
    restore: bool = True,
) -> dict:
    """Read a multi-page scrollable pane (terminal, Copilot TUI, browser, editor) without sending keystrokes.

    Moves the pointer to (x, y), OCRs each viewport (optionally cropped to `region`), scrolls
    `notches_per_step` wheel detents per step for up to `steps` steps (stopping early when clamped
    for 2 consecutive steps), deduplicates overlapping lines into top-to-bottom reading order, and
    scrolls back to restore the original position when `restore=True`."""
    steps = max(1, min(40, steps))
    notches_per_step = max(1, min(20, notches_per_step))
    await client.mouse_move(x, y)

    pages: list[list[str]] = []
    seen_norms: set[str] = set()
    stalled_steps = 0
    total_scrolled = 0
    prev_sha: str | None = None

    for step_idx in range(steps + 1):
        if step_idx > 0:
            await client.scroll(notches_per_step, direction)
            total_scrolled += notches_per_step
            if settle_ms > 0:
                await asyncio.sleep(min(5000, settle_ms) / 1000.0)
        img, info = await _snap()
        sha = info["frame_sha"]
        if step_idx > 0 and sha == prev_sha:
            break
        prev_sha = sha
        tree, _ = _run_or_get_tree(img, sha, region=region, force_refresh=False)
        rows = extract_text_rows(tree)
        new_in_step = [ln for ln in rows if _norm(ln) and _norm(ln) not in seen_norms]
        for ln in new_in_step:
            seen_norms.add(_norm(ln))
        pages.append(rows)
        if step_idx > 0:
            if not new_in_step:
                stalled_steps += 1
                if stalled_steps >= 2:
                    break
            else:
                stalled_steps = 0

    if restore and total_scrolled > 0:
        opp = "down" if direction == "up" else "up"
        await client.scroll(total_scrolled, opp)

    # Order pages top-to-bottom in document order, deduplicating overlapping lines
    ordered_pages = list(reversed(pages)) if direction == "up" else pages
    deduped: list[str] = []
    emitted: set[str] = set()
    for pg in ordered_pages:
        for ln in pg:
            key_n = _norm(ln)
            if not key_n or key_n in emitted:
                continue
            emitted.add(key_n)
            deduped.append(ln)

    return {
        "ok": True,
        "direction": direction,
        "pages_captured": len(pages),
        "total_notches_scrolled": total_scrolled,
        "restored": bool(restore and total_scrolled > 0),
        "line_count": len(deduped),
        "lines": deduped,
        "text": "\n".join(deduped),
    }


# ---------------------------------------------------------------- keyboard

@mcp.tool()
async def key(
    name: str,
    action: Literal["tap", "down", "up"] = "tap",
    repeat: int = 1,
    inspect_after: Literal["none", "lines", "diff", "text"] = "none",
    return_screenshot: bool = False,
    settle_ms: int = 250,
    region: list[int] | None = None,
) -> Any:
    """Send a keyboard key by KeyboardEvent.code name (MetaLeft, KeyA, Enter, F5,
    Escape, Numpad1, Space, ArrowLeft, ControlLeft, ShiftLeft, AltLeft, ...).

    MetaLeft tap opens the Start menu. Pass repeat>1 to tap multiple times (e.g. ArrowDown x5)."""
    await _capture_pre_baseline(inspect_after, region)
    reps = max(1, min(50, repeat)) if action == "tap" else 1
    if action == "tap":
        for i in range(reps):
            if i:
                await asyncio.sleep(0.02)
            await client.key_tap(name)
    elif action == "down":
        await client.key_down(name)
    else:
        await client.key_up(name)
    res: dict[str, Any] = {"ok": True, "key": name, "action": action}
    if reps > 1:
        res["repeat"] = reps
    return await _with_post_action(
        res,
        inspect_after=inspect_after,
        return_screenshot=return_screenshot,
        settle_ms=settle_ms,
        region=region,
    )


@mcp.tool()
async def key_combo(
    keys: list[str],
    inspect_after: Literal["none", "lines", "diff", "text"] = "none",
    return_screenshot: bool = False,
    settle_ms: int = 250,
    region: list[int] | None = None,
) -> Any:
    """Send a chorded shortcut, e.g. ["MetaLeft", "KeyE"] (Win+E File Explorer),
    ["ControlLeft", "ShiftLeft", "KeyT"], ["AltLeft", "Tab"]. Pressed in order,
    released in reverse."""
    await _capture_pre_baseline(inspect_after, region)
    await client.key_combo(keys)
    return await _with_post_action(
        {"ok": True, "keys": keys},
        inspect_after=inspect_after,
        return_screenshot=return_screenshot,
        settle_ms=settle_ms,
        region=region,
    )


@mcp.tool()
async def type_text(
    text: str,
    delay_ms: int = 4,
    press_enter: bool = False,
    inspect_after: Literal["none", "lines", "diff", "text"] = "none",
    return_screenshot: bool = False,
    settle_ms: int = 250,
    region: list[int] | None = None,
) -> Any:
    """Type text on the target as real keystrokes via the target's own OS keyboard
    layout (GLKVM_KEYBOARD_LAYOUT selects the layout table, default 'uk', with Shift
    coalesced across runs). Set press_enter=True to tap Enter after typing, and
    inspect_after='diff'|'lines'|'text' or return_screenshot=True to observe the result
    in the same round-trip."""
    await _capture_pre_baseline(inspect_after, region)
    await client.type_text(text, max(0, delay_ms) / 1000)
    if press_enter:
        await client.key_tap("Enter")
    res: dict[str, Any] = {"ok": True, "typed": text, "chars": len(text), "layout": LAYOUT}
    if press_enter:
        res["press_enter"] = True
    return await _with_post_action(
        res,
        inspect_after=inspect_after,
        return_screenshot=return_screenshot,
        settle_ms=settle_ms,
        region=region,
    )


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
async def video_state() -> dict:
    """HDMI link state on the KVM: signal, resolution, capture fps. Cheap
    pre-check before screenshotting (signal false = the target's GPU has dropped
    its output; screenshot() then re-emits EDID automatically)."""
    return {"ok": True, **await client.video_state()}


@mcp.tool()
async def edid_prod(timeout_s: int = 15) -> dict:
    """Re-emit the current EDID preset (HPD pulse) to wake a target whose GPU
    dropped its HDMI output - the usual symptom after a KVM reboot: Windows
    still shows 'duplicate' on but the KVM sees no signal, and screenshots
    hang/fail (screenshot() auto-prods once in that case). Same action as
    flicking the EDID setting in the GL web UI. Harmless while the signal is
    healthy; polls up to timeout_s seconds for the signal to return."""
    return await client.edid_prod(timeout_s=timeout_s)


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
    """Target display resolution, active non-pillarboxed bounds, and fresh frame hash."""
    _, info = await _snap()
    return {"ok": True, **info}


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