# glkvm-mcp: MCP server for a GL.iNet GLKVM (PiKVM kvmd fork) over HTTPS.
#
# Empirically verified protocol (GLKVM kvmd 4.82 fork, firmware quirk noted):
#   auth     POST /api/auth/login   form user=<u>&passwd=<p> -> {"ok":true,"result":{"token":...}}
#            endpoint auth: ?auth_token=<token> query param (also "token" header)
#   mouse    POST /api/hid/events/send_mouse_move?to_x=&to_y=        device int16 abs coords
#            POST /api/hid/events/send_mouse_button?button=left      no state -> press+release
#            POST /api/hid/events/send_mouse_wheel?delta_x=&delta_y= int8, -y scrolls down
#   keyboard POST /api/hid/events/send_key?key=KeyA&state=1|0        KeyboardEvent.code names
#            POST /api/hid/print                                     body=utf-8 text, server-side keysym typing
#            POST /api/hid/events/send_shortcut?keys=MetaLeft,KeyE   press-all, release-reverse
#   video    GET  /streamer/snapshot                                 live JPEG of target display
#   state    GET  /api/hid, GET /api/hid/inactivity
#
# Absolute-mouse calibration at 2560x1440 (measured, +-1px):
#   x_px = 0.05*W + 0.90*W*(v_x+32768)/65535   (GL firmware maps only the middle 90% width)
#   y_px = 1.00*H*(v_y+32768)/65535            (full height)
#   inverse: v_x = (x_px-0.05*W)*65535/(0.90*W)-32768 ; v_y = y_px*65535/H-32768

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import ssl
import time
from typing import Any

import httpx
from PIL import Image as PILImage

HOST = os.environ.get("GLKVM_HOST", "glkvm.lan")
USER = os.environ.get("GLKVM_USER", "admin")
PASSWORD = os.environ.get("GLKVM_PASSWORD", "")
BASE = os.environ.get("GLKVM_BASE", f"https://{HOST}")

_TOKEN_FILE = os.path.expanduser(f"~/.cache/glkvm-mcp/{HOST}.token.json")

# Empirical GL absolute-mouse mapping constants (fractions of frame size).
_X_OFF_FRAC = 0.05
_X_SPAN_FRAC = 0.90


def v_from_px_x(x: float, w: int) -> int:
    return max(-32768, min(32767, round((x - _X_OFF_FRAC * w) * 65535 / (_X_SPAN_FRAC * w)) - 32768))


def v_from_px_y(y: float, h: int) -> int:
    return max(-32768, min(32767, round(y * 65535 / h) - 32768))


def px_from_v(vx: int, vy: int, w: int, h: int) -> tuple[float, float]:
    x = _X_OFF_FRAC * w + _X_SPAN_FRAC * w * (vx + 32768) / 65535
    y = h * (vy + 32768) / 65535
    return (x, y)


class GlkvmError(RuntimeError):
    pass


class GlkvmClient:
    """Async HTTP driver for the GLKVM kvmd API. All coordinates are target-screen px."""

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        self._ssl = ssl._create_unverified_context()
        self._token: str | None = None
        self._login_lock = asyncio.Lock()
        self._w = 2560
        self._h = 1440
        self.last_input_ts = 0.0  # time of last HID event WE sent
        self._motion_ref: tuple[float, list[float]] | None = None

    # -- plumbing ----------------------------------------------------------

    async def client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=BASE, verify=self._ssl, timeout=httpx.Timeout(20.0),
            )
        return self._http

    async def login(self, force: bool = False) -> str:
        if not PASSWORD:
            raise GlkvmError("set GLKVM_PASSWORD (device web-admin password) in the server env")
        async with self._login_lock:
            if not force and self._token:
                return self._token
            if not force:
                tok = self._load_token_file()
                if tok:
                    self._token = tok
                    return tok
            http = await self.client()
            r = await http.post("/api/auth/login", data={"user": USER, "passwd": PASSWORD})
            r.raise_for_status()
            j = r.json()
            if not j.get("ok"):
                raise GlkvmError(f"login failed: {j.get('message') or j}")
            self._token = j["result"]["token"]
            self._save_token_file(self._token)
            return self._token

    def _token_file(self) -> str:
        d = os.path.expanduser("~/.cache/glkvm-mcp")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{HOST}.token.json")

    def _load_token_file(self) -> str | None:
        try:
            with open(self._token_file()) as f:
                j = json.load(f)
            tok = j.get("token")
            return tok if isinstance(tok, str) and len(tok) >= 32 else None
        except Exception:
            return None

    def _save_token_file(self, tok: str) -> None:
        try:
            with open(self._token_file(), "w") as f:
                json.dump({"token": tok, "ts": time.time()}, f)
        except Exception:
            pass

    async def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        raw_content: bytes | None = None,
        _retry: bool = True,
    ) -> httpx.Response:
        tok = await self.login()
        http = await self.client()
        q = dict(query or {})
        q.setdefault("auth_token", tok)
        kw: dict[str, Any] = {}
        if data is not None:
            kw["data"] = data
        if raw_content is not None:
            kw["content"] = raw_content
        r = await http.request(method, path, params=q, headers={"token": tok}, **kw)
        if r.status_code == 401 and _retry:
            await self.login(force=True)
            return await self.request(method, path, query=query, data=data,
                                      raw_content=raw_content, _retry=False)
        if r.status_code == 403 and _retry and "ForbiddenError" in r.text:
            # GL fork: an unknown but format-valid token returns 403
            # ForbiddenError (kvmd restart invalidates the cached token;
            # genuinely denied permissions come back 401 or 400 instead).
            await self.login(force=True)
            return await self.request(method, path, query=query, data=data,
                                      raw_content=raw_content, _retry=False)
        if r.status_code >= 400:
            snippet = r.text[:200]
            raise GlkvmError(f"{method} {path} -> {r.status_code}: {snippet}")
        return r

    async def get_json(self, path: str, *, query: dict[str, Any] | None = None) -> Any:
        r = await self.request("GET", path, query=query)
        return r.json()

    async def post_ok(self, path: str, *, query: dict[str, Any] | None = None,
                      raw_content: bytes | None = None) -> dict:
        r = await self.request("POST", path, query=query, raw_content=raw_content)
        try:
            j = r.json()
        except Exception:
            j = {"ok": r.is_success}
        if isinstance(j, dict) and j.get("ok") is False:
            raise GlkvmError(f"{path}: {j.get('message') or j}")
        return j if isinstance(j, dict) else {"ok": True}

    # -- keyboard ----------------------------------------------------------

    async def key_down(self, key: str) -> None:
        await self.post_ok("/api/hid/events/send_key", query={"key": key, "state": 1})
        self.last_input_ts = time.time()

    async def key_up(self, key: str) -> None:
        await self.post_ok("/api/hid/events/send_key", query={"key": key, "state": 0})
        self.last_input_ts = time.time()

    async def key_tap(self, key: str, hold_s: float = 0.03) -> None:
        await self.key_down(key)
        if hold_s > 0:
            await asyncio.sleep(hold_s)
        await self.key_up(key)

    async def key_combo(self, keys: list[str], hold_s: float = 0.06) -> None:
        """Press all keys (e.g. MetaLeft+KeyE), release in reverse."""
        for k in keys:
            await self.key_down(k)
            await asyncio.sleep(0.02)
        if hold_s > 0:
            await asyncio.sleep(hold_s)
        for k in reversed(keys):
            await self.key_up(k)
            await asyncio.sleep(0.02)

    async def type_text(self, text: str, delay: float = 0.02) -> None:
        """Unicode typing via kvmd's server-side keysym printer."""
        if not text:
            return
        chunks = [text[i:i + 900] for i in range(0, len(text), 900)]
        for c in chunks:
            await self.post_ok("/api/hid/print", query={"delay": delay}, raw_content=c.encode("utf-8"))
        self.last_input_ts = time.time()

    # -- mouse ---------------------------------------------------------------

    async def mouse_move(self, x: float, y: float) -> tuple[int, int]:
        vx, vy = v_from_px_x(x, self._w), v_from_px_y(y, self._h)
        await self.post_ok("/api/hid/events/send_mouse_move",
                           query={"to_x": vx, "to_y": vy})
        self.last_input_ts = time.time()
        return (vx, vy)

    async def mouse_button(self, button: str, state: bool) -> None:
        await self.post_ok("/api/hid/events/send_mouse_button",
                           query={"button": button, "state": 1 if state else 0})
        self.last_input_ts = time.time()

    async def mouse_click(self, x: float | None = None, y: float | None = None,
                          button: str = "left") -> None:
        if x is not None or y is not None:
            await self.mouse_move(x, y)
            await asyncio.sleep(0.02)
        await self.post_ok("/api/hid/events/send_mouse_button", query={"button": button})
        self.last_input_ts = time.time()

    async def scroll(self, notches: int = 3, direction: str = "down") -> None:
        step = 3  # measured: device y=-3 == one wheel notch down
        per = {"down": (0, -step), "up": (0, step), "right": (step, 0), "left": (-step, 0)}[direction]
        for _ in range(abs(notches)):
            await self.post_ok("/api/hid/events/send_mouse_wheel",
                               query={"delta_x": per[0], "delta_y": per[1]})
            await asyncio.sleep(0.03)
        self.last_input_ts = time.time()

    # -- MSD / mass-storage file transfer -------------------------------------

    async def _authed_params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        tok = await self.login()
        q = {"auth_token": tok}
        q.update(extra or {})
        return q

    async def msd_write_image(self, local_path: str, image: str) -> int:
        """Upload a local disk image into MSD storage (kvmd stores disk images,
        not files - wrap files in a FAT image first). Streams the file in 1 MiB
        chunks. Returns bytes written."""
        size = os.path.getsize(local_path)
        if size <= 0:
            raise GlkvmError(f"empty image file: {local_path}")
        tok = await self.login()
        http = await self.client()
        q = await self._authed_params({"image": image, "remove_incomplete": True})
        # manual Content-Length because generator content has no length
        headers = {"token": tok, "Content-Length": str(size)}

        async def chunks():
            f = await asyncio.to_thread(open, local_path, "rb")
            try:
                while True:
                    b = await asyncio.to_thread(f.read, 1 << 20)
                    if not b:
                        break
                    yield b
            finally:
                await asyncio.to_thread(f.close)

        r = await http.request("POST", "/api/msd/write", params=q, headers=headers,
                               content=chunks())
        if r.status_code == 401:
            await self.login(force=True)
            return await self.msd_write_image(local_path, image)
        if r.status_code >= 400:
            raise GlkvmError(f"msd/write -> {r.status_code}: {r.text[:200]}")
        j = r.json()
        written = int(j.get("result", {}).get("image", {}).get("written", -1))
        if written != size:
            raise GlkvmError(f"msd/write wrote {written} of {size}")
        return written

    async def msd_download_image(self, image: str, out_path: str,
                                 compress: str = "zstd") -> int:
        """Stream an MSD image back to a local file ("reversed" transfer).
        compress: "" (raw), "zstd" or "lzma". Returns bytes written."""
        tok = await self.login()
        http = await self.client()
        q = await self._authed_params({"image": image, "compress": compress})
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        total = 0
        async with http.stream("GET", "/api/msd/read", params=q,
                               headers={"token": tok}) as r:
            if r.status_code == 401:
                await self.login(force=True)
                return await self.msd_download_image(image, out_path, compress)
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")[:200]
                raise GlkvmError(f"msd/read -> {r.status_code}: {body}")
            with open(out_path, "wb") as f:
                async for chunk in r.aiter_bytes(1 << 20):
                    f.write(chunk)
                    total += len(chunk)
        return total

    async def msd_connect(self, image: str | None = None, cdrom: bool = False,
                          rw: bool = False, connected: bool = True) -> dict:
        """Attach a stored image to the target as a USB drive (cdrom=False),
        or detach (connected=False). GL fork has both a cdrom drive and an rw
        partition drive; params land on the active one."""
        if image is None or not connected:
            return await self.post_ok("/api/msd/set_connected",
                                      query={"connected": 1 if connected else 0})
        await self.post_ok("/api/msd/set_params",
                           query={"image": image, "cdrom": 1 if cdrom else 0,
                                  "rw": 1 if rw else 0})
        return await self.post_ok("/api/msd/set_connected", query={"connected": 1})

    async def msd_state(self) -> dict:
        st = await self.get_json("/api/msd")
        return st.get("result", st) if isinstance(st, dict) else st

    async def msd_remove(self, image: str) -> None:
        await self.post_ok("/api/msd/remove", query={"image": image})
        # note: GL fork's state listing may briefly keep showing the removed
        # image (state-cache race observed); re-check after ~1s if needed.

    async def snapshot(self) -> tuple[PILImage.Image, str]:
        r = await self.request("GET", "/streamer/snapshot")
        img = PILImage.open(io.BytesIO(r.content))
        img.load()
        if (img.width, img.height) != (self._w, self._h):
            self._w, self._h = img.width, img.height
        sha = hashlib.sha256(r.content).hexdigest()[:12]
        return (img, sha)

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._w, self._h)

    # -- state / activity --------------------------------------------------------

    async def inactivity(self) -> float:
        j = await self.get_json("/api/hid/inactivity")
        return float(j.get("inactivity", -1))

    async def hid_state(self) -> dict:
        j = await self.get_json("/api/hid")
        return j.get("result", j) if isinstance(j, dict) else j

    async def motion_probe(self, span_s: float = 1.2, n: int = 3) -> dict:
        """Fraction of changed pixels between consecutive downscaled frames."""
        import numpy as np

        frames: list[np.ndarray] = []
        t0 = time.time()
        for _ in range(n):
            img, _ = await self.snapshot()
            g = img.convert("L").resize((160, 90))
            frames.append(np.asarray(g, dtype=np.int16))
            if len(frames) < n:
                await asyncio.sleep(max(0.0, (span_s / (n - 1)) - 0.05))
        diffs = []
        for a, b in zip(frames, frames[1:]):
            diffs.append(float((abs(b - a) > 24).mean()))
        return {"span_s": round(time.time() - t0, 2), "diffs": [round(d, 5) for d in diffs]}

    async def interference_check(self) -> dict:
        """Detect concurrent human input.

        Preferred signal is kvmd's HID inactivity counter (any input - ours, a
        human's physical KVM input, or another kvmd client like the browser UI).
        The GL firmware 404s that endpoint, so the fallback is screen motion:
        frames changing while this tool was idle (quiet) imply an active person
        or another client on the target."""
        inact: float | None = None
        try:
            inact = await self.inactivity()
        except Exception:
            pass
        quiet = time.time() - self.last_input_ts
        out: dict[str, Any] = {
            "hid_inactivity_s": round(inact, 1) if inact is not None and inact >= 0 else None,
            "mcp_quiet_s": round(quiet, 1),
        }
        if inact is not None:
            foreign = quiet > 3.0 and inact < 3.0
            out["verdict"] = ("foreign HID input detected (not from this tool)" if foreign
                              else "quiet (no input outside this tool in the last few seconds)")
            out["warning"] = foreign
            return out
        # screen-motion fallback
        motion = await self.motion_probe(span_s=1.0, n=2)
        top = max(motion["diffs"], default=0.0)
        out["screen"] = {"changed_frac": round(top, 4)}
        # screen visibly moving while this tool has been idle >3s -> someone else
        foreign = quiet > 3.0 and top > 0.004  # >0.4% of a 160x90 thumbnail changed
        if foreign:
            out["verdict"] = ("target screen is moving while this tool is idle - a human "
                              "or another KVM client may be working on it")
        elif quiet <= 3.0:
            out["verdict"] = "recent tool input; motion is probably ours"
        else:
            out["verdict"] = "quiet (screen static while this tool is idle)"
        out["warning"] = foreign
        return out