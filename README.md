# glkvm-mcp

MCP server for the GLKVM KVM session: accurate mouse, complete keyboard, OCR element tree, human-interference detection. It drives a GL.iNet GLKVM (PiKVM kvmd fork) over its HTTPS API, with no agent installed on the target; the target sees only an HID composite device and the HDMI input.

Built and verified against a GL.iNet RM10 running kvmd 4.82 (GL fork). Other GLKVM models on the same firmware should behave the same. Stock upstream PiKVM is untested: the driver re-logins on the GL fork's stale-token 403 behaviour and falls back to screen-motion probing when `/api/hid/inactivity` 404s, none of which runs against stock kvmd.

## Requirements

- Python 3.12+ with uv
- A GLKVM device reachable over HTTPS, and its web UI password
- OCR runs locally: rapidocr-onnxruntime on CPU (frames are not sent anywhere)

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GLKVM_HOST` | `glkvm.lan` | Device hostname |
| `GLKVM_USER` | `admin` | Web UI user |
| `GLKVM_PASSWORD` | unset | Web UI password, required |
| `GLKVM_BASE` | `https://<GLKVM_HOST>` | Full base URL override |
| `GLKVM_ALLOW_MSD` | unset | Set to `1`/`true`/`yes` to enable the `msd_*` tools |
| `GLKVM_KEYBOARD_LAYOUT` | `uk` | Target OS keyboard layout for `type_text` (`uk` or `us`) |

Setup and registration in Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "glkvm": {
      "command": "uv",
      "args": ["run", "python", "-m", "glkvm_mcp.server"],
      "env": { "GLKVM_PASSWORD": "your-device-password" }
    }
  }
}
```

## Tools

Screen:

- `screenshot` - live frame (default `max_width=1920`, so 1080p frames stay 1:1 with click coordinates) plus structured resolution, scale, and active pillarbox bounds; optional `region=[x,y,w,h]` crop
- `screen_elements` - OCR of the current frame in full-screen pixel coordinates; supports `mode="lines" | "text" | "find" | "diff" | "tree"`, `region=[x,y,w,h]` ROI cropping, and `include_image=False` to avoid redundant image transfers
- `read_scrollable` - multi-step wheel-scroll + RapidOCR line deduplicator for reading terminal/TUI/browser scrollback without sending keystrokes (restores scroll position afterwards)
- `snapshot_info` - display resolution, active non-pillarboxed horizontal bounds, and a frame hash to test freshness
- `edid_prod` - re-apply the current EDID preset, which re-asserts EDID/HPD on the wire; wakes a target whose GPU dropped its HDMI output after a KVM reboot (`_snap()` does that once automatically across vision tools when it hits a no-signal error)
- `video_state` - HDMI signal, resolution and capture fps, without grabbing a frame

Input (coordinates are target-screen pixels; `click_at`, `click_text`, `scroll`, `key`, `key_combo`, and `type_text` accept `inspect_after="diff"|"lines"|"text"` and `return_screenshot=True` for single-round-trip action + observation):

- `click_at`, `click_text`, `move_mouse`, `mouse_relative`, `drag`, `scroll`
- `key`, `key_combo`, `type_text` - keys are KeyboardEvent.code names, so `MetaLeft` opens the target's Start menu; `type_text` types printable ASCII as real keystrokes using the target's own OS layout (`GLKVM_KEYBOARD_LAYOUT`, default `uk`, with `ShiftLeft` coalesced across shifted runs and optional `press_enter=True`), falling back to the kvmd keysym printer (US layout) for accented chars and other Unicode

State and safety:

- `human_activity` - warns when input the tool did not produce is happening (keyboard/mouse inactivity counter, or screen motion as fallback)
- `kvmd_api` - raw GET/POST passthrough for anything the driver does not wrap

File transfer (opt-in via `GLKVM_ALLOW_MSD=1`, see below):

- `msd_state`, `msd_push_files`, `msd_pull_files`, `msd_connect`, `msd_disconnect`, `msd_remove_image`

The `msd_*` tools use the KVM's emulated USB mass-storage drive: push wraps files in a FAT image and attaches it as a USB disk on the target; pull streams the image back after the target writes files to it. They are disabled by default because attaching USB storage is visible to the target (and to any EDP/security tooling on it), while the HID + HDMI path stays quiet. For managed targets, keep the MSD tools off.

## Accuracy notes

Absolute mouse mapping uses the USB HID digitizer's `[-32768..32767]` range across the active target desktop (`[active_left_px, active_left_px + active_width_px]`). On a native 16:9 target `active_left_px = 0` and `active_width_px = width`; when a 16:10 laptop display is mirrored/pillarboxed inside a 16:9 HDMI capture frame (e.g. `1728x1080` centered inside `1920x1080` with 96px studio-black bars on each side), `snapshot()` detects the pillarbox bars automatically with 2px hysteresis. OCR boxes are pixel-accurate for line boxes, a few px for word boxes.

Verified live: keyboard (keys, combos, layout-aware typing on a UK Windows target, including `\`, `|`, `@`, `"`, `#`, `~`, and Unicode printing over `/api/hid/print`), mouse absolute/relative/wheel/drag, caps-lock LED echo probe, snapshot, OCR tree, MSD push/pull byte-exact roundtrip, `edid_prod` on a healthy HDMI signal (device replies "EDID data has been written and applied", capture fps dips for a few seconds, signal never drops). Not exercised live: `msd_connect`/`msd_disconnect` attach/detach (deliberately - it injects a USB-drive event into the session in use), horizontal wheel direction, `human_activity` against a genuinely concurrent human operator, and `edid_prod` recovering a genuinely no-signal target (first real drop will show it).

## License

AGPL-3.0-or-later, see `LICENSE`.