from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image as PILImage

from glkvm_mcp import kvmd, server, vision


def _make_frame(w: int, h: int, bar_px: int = 0, bar_rgb=(16, 16, 16), body_rgb=(40, 44, 52)) -> PILImage.Image:
    arr = np.full((h, w, 3), body_rgb, dtype=np.uint8)
    if bar_px > 0:
        arr[:, :bar_px] = bar_rgb
        arr[:, w - bar_px:] = bar_rgb
    return PILImage.fromarray(arr, mode="RGB")


class PillarboxAndGeometryTests(unittest.TestCase):
    def test_native_16_9_uses_full_frame_width(self):
        img = _make_frame(2560, 1440, bar_px=0)
        left, active_w = kvmd.detect_horizontal_pillarbox(img)
        self.assertEqual((left, active_w), (0, 2560))
        self.assertEqual(kvmd.v_from_px_x(0, 2560, left, active_w), -32768)
        self.assertGreater(kvmd.v_from_px_x(64, 2560, left, active_w), -32768)
        self.assertEqual(kvmd.v_from_px_x(2560, 2560, left, active_w), 32767)

    def test_16_10_pillarboxed_in_1080p_detects_studio_black_bars(self):
        img = _make_frame(1920, 1080, bar_px=96, bar_rgb=(16, 16, 16))
        left, active_w = kvmd.detect_horizontal_pillarbox(img)
        self.assertEqual((left, active_w), (96, 1728))
        self.assertEqual(kvmd.v_from_px_x(96, 1920, left, active_w), -32768)
        self.assertEqual(kvmd.v_from_px_x(1824, 1920, left, active_w), 32767)
        rx, ry = kvmd.px_from_v(0, 0, 1920, 1080, left, active_w)
        self.assertAlmostEqual(rx, 960.0, delta=1.0)
        self.assertAlmostEqual(ry, 540.0, delta=1.0)

    def test_pillarbox_hysteresis_ignores_1px_jpeg_ringing(self):
        c = kvmd.GlkvmClient()
        c._update_frame_geometry(_make_frame(1920, 1080, bar_px=96))
        self.assertEqual(c.active_bounds, (96, 1728))
        # 1px JPEG ringing on next tick must not shift active_bounds
        c._update_frame_geometry(_make_frame(1920, 1080, bar_px=95))
        self.assertEqual(c.active_bounds, (96, 1728))
        # Genuine transition to full-width 16:9 updates immediately
        c._update_frame_geometry(_make_frame(1920, 1080, bar_px=0))
        self.assertEqual(c.active_bounds, (0, 1920))


class ClientBehaviourTests(unittest.IsolatedAsyncioTestCase):
    async def test_type_text_coalesces_shift_across_uppercase_run(self):
        c = kvmd.GlkvmClient()
        events: list[tuple[str, int]] = []

        async def fake_post_ok(path: str, *, query=None, raw_content=None):
            if path == "/api/hid/events/send_key":
                events.append((query["key"], query["state"]))
            return {"ok": True}

        c.post_ok = fake_post_ok  # type: ignore[method-assign]
        await c.type_text("AZa", delay=0.0)
        self.assertEqual(
            events,
            [
                ("ShiftLeft", 1),
                ("KeyA", 1),
                ("KeyA", 0),
                ("KeyZ", 1),
                ("KeyZ", 0),
                ("ShiftLeft", 0),
                ("KeyA", 1),
                ("KeyA", 0),
            ],
        )

    async def test_inactivity_404_is_cached_across_calls(self):
        c = kvmd.GlkvmClient()
        inact_calls = 0

        async def fake_inactivity():
            nonlocal inact_calls
            inact_calls += 1
            raise kvmd.GlkvmError("GET /api/hid/inactivity -> 404: Not Found")

        c.inactivity = fake_inactivity  # type: ignore[method-assign]
        c.motion_probe = AsyncMock(return_value={"span_s": 0.1, "diffs": [0.0]})  # type: ignore[method-assign]
        await c.interference_check()
        await c.interference_check()
        self.assertEqual(inact_calls, 1)


class ServerStreamliningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._tree_cache.clear()
        server._last_lines_snapshot.clear()

    async def test_click_at_skips_slow_interference_probe_by_default(self):
        with (
            patch.object(server.client, "interference_check", new=AsyncMock()) as mock_inter,
            patch.object(server.client, "mouse_click", new=AsyncMock()) as mock_click,
        ):
            res = await server.click_at(x=500, y=400)
            self.assertTrue(res["ok"])
            mock_click.assert_awaited_once_with(500, 400, "left")
            mock_inter.assert_not_awaited()

    async def test_screenshot_returns_1_to_1_1080p_and_structured_meta(self):
        img = _make_frame(1920, 1080, bar_px=96)

        async def fake_snap():
            server.client._update_frame_geometry(img)
            return img, "abc123def456"

        with patch.object(server.client, "snapshot", new=fake_snap):
            out = await server.screenshot()
            meta = out.structured_content
            self.assertEqual(meta["width"], 1920)
            self.assertEqual(meta["height"], 1080)
            self.assertEqual(meta["image_width"], 1920)
            self.assertEqual(meta["image_height"], 1080)
            self.assertEqual(meta["scale"], 1.0)
            self.assertEqual(meta["active_left_px"], 96)
            self.assertEqual(meta["active_width_px"], 1728)

    async def test_screen_elements_modes_region_offset_and_diff(self):
        img = _make_frame(1920, 1080, bar_px=96)
        ocr_frame_1 = [
            [[[10, 20], [110, 20], [110, 40], [10, 40]], "Apply changes", 0.98],
            [[[10, 60], [90, 60], [90, 80], [10, 80]], "Cancel", 0.97],
        ]
        ocr_frame_2 = [
            [[[10, 20], [110, 20], [110, 40], [10, 40]], "Apply changes", 0.98],
            [[[10, 95], [140, 95], [140, 115], [10, 115]], "Saved ok", 0.99],
        ]

        with (
            patch.object(server.client, "snapshot", new=AsyncMock(side_effect=[(img, "sha001"), (img, "sha002")])),
            patch.object(server.ocr, "run", side_effect=[(ocr_frame_1, 12.5), (ocr_frame_2, 11.0)]),
        ):
            r1 = await server.screen_elements(
                mode="lines", region=[300, 400, 500, 200], include_image=False
            )
            d1 = r1.structured_content
            self.assertEqual(len(r1.content), 1)  # JSON only when include_image=False
            self.assertEqual(d1["lines"][0]["text"], "Apply changes")
            # Region offset [300, 400] added back into full-screen coordinates
            self.assertEqual(d1["lines"][0]["bbox"], [310, 420, 100, 20])

            r2 = await server.screen_elements(
                mode="diff", region=[300, 400, 500, 200], include_image=False
            )
            d2 = r2.structured_content
            self.assertTrue(d2["changed"])
            self.assertEqual([l["text"] for l in d2["added_lines"]], ["Saved ok"])
            self.assertEqual([l["text"] for l in d2["removed_lines"]], ["Cancel"])

    async def test_click_text_anchor_offset_and_post_inspect(self):
        img = _make_frame(1920, 1080, bar_px=0)
        ocr_before = [[[[200, 100], [300, 100], [300, 130], [200, 130]], "Duplicate", 0.99]]
        ocr_after = [
            [[[200, 100], [300, 100], [300, 130], [200, 130]], "Duplicate", 0.99],
            [[[200, 150], [340, 150], [340, 180], [200, 180]], "Keep changes", 0.96],
        ]
        with (
            patch.object(server.client, "snapshot", new=AsyncMock(side_effect=[(img, "s1"), (img, "s2")])),
            patch.object(server.ocr, "run", side_effect=[(ocr_before, 10.0), (ocr_after, 10.0)]),
            patch.object(server.client, "mouse_click", new=AsyncMock()) as mock_click,
        ):
            res = await server.click_text(
                query="Duplicate", anchor="right", dx=12, dy=0, inspect_after="diff", settle_ms=0
            )
            self.assertTrue(res["ok"])
            self.assertEqual(res["clicked"], [312, 115])
            mock_click.assert_awaited_once_with(312, 115, "left")
            self.assertEqual([l["text"] for l in res["inspection"]["added_lines"]], ["Keep changes"])

    async def test_read_scrollable_dedupes_and_restores_scroll(self):
        img = _make_frame(1920, 1080, bar_px=0)
        frames = [
            [(img, "f1"), ([[[[10, 10], [80, 10], [80, 30], [10, 30]], "line 3", 0.99],
                           [[[10, 40], [80, 40], [80, 60], [10, 60]], "line 4", 0.99]], 8.0)],
            [(img, "f2"), ([[[[10, 10], [80, 10], [80, 30], [10, 30]], "line 1", 0.99],
                           [[[10, 40], [80, 40], [80, 60], [10, 60]], "line 3", 0.99]], 8.0)],
            [(img, "f2"), ([[[[10, 10], [80, 10], [80, 30], [10, 30]], "line 1", 0.99],
                           [[[10, 40], [80, 40], [80, 60], [10, 60]], "line 3", 0.99]], 8.0)],
        ]
        snaps = [f[0] for f in frames]
        ocrs = [f[1] for f in frames]
        scroll_calls: list[tuple[int, str]] = []

        async def fake_scroll(notches: int = 3, direction: str = "down"):
            scroll_calls.append((notches, direction))

        with (
            patch.object(server.client, "snapshot", new=AsyncMock(side_effect=snaps)),
            patch.object(server.ocr, "run", side_effect=ocrs),
            patch.object(server.client, "mouse_move", new=AsyncMock()),
            patch.object(server.client, "scroll", new=fake_scroll),
        ):
            out = await server.read_scrollable(
                x=600, y=500, direction="up", steps=4, notches_per_step=6, settle_ms=0, restore=True
            )
            self.assertTrue(out["ok"])
            # Ordered top-to-bottom when scrolling up: older lines above newer lines
            self.assertEqual(out["lines"], ["line 1", "line 3", "line 4"])
            # Scrolled up twice (12 notches) before clamping, then restored 12 notches down
            up_total = sum(n for n, d in scroll_calls if d == "up")
            down_total = sum(n for n, d in scroll_calls if d == "down")
            self.assertEqual(up_total, down_total)


if __name__ == "__main__":
    unittest.main()
