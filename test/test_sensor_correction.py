import os
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from lfstitcher import SimpleStitcher


class TestSensorCorrection(unittest.TestCase):
    def _make_synthetic_stack(self, num_frames=8, h=240, w=320):
        """Create shifted textured frames with stationary vignetting, dust, and brightness shifts."""
        rng = np.random.default_rng(42)
        # Large synthetic specimen canvas
        canvas_h, canvas_w = h + 120, w + 400
        specimen = np.full((canvas_h, canvas_w, 3), 220, dtype=np.uint8)
        for _ in range(180):
            cx = int(rng.integers(40, canvas_w - 40))
            cy = int(rng.integers(20, canvas_h - 20))
            rad = int(rng.integers(6, 18))
            color = tuple(int(c) for c in rng.integers(40, 140, size=3))
            cv2.circle(specimen, (cx, cy), rad, color, -1)

        # Stationary illumination gradient / vignetting
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        illum = 1.0 - 0.30 * (((xx - w * 0.4) / w) ** 2 + ((yy - h * 0.5) / h) ** 2)
        # Stationary semi-transparent smudge
        smudge = 1.0 - 0.18 * np.exp(-(((xx - 90) ** 2 + (yy - 80) ** 2) / (2 * 18.0 ** 2)))
        # Stationary opaque dust spot
        dust_dist = np.sqrt((xx - 220) ** 2 + (yy - 140) ** 2)
        dust = np.where(dust_dist <= 4.0, 0.15, 1.0).astype(np.float32)

        sensor_field = (illum * smudge * dust)[:, :, None]

        frames = []
        brightness_scales = np.linspace(0.85, 1.10, num_frames)
        for i in range(num_frames):
            x0 = i * 40
            y0 = 40 + (i % 3) * 10
            crop = specimen[y0:y0 + h, x0:x0 + w].astype(np.float32)
            frame = np.clip(crop * sensor_field * brightness_scales[i], 0, 255).astype(np.uint8)
            frames.append(frame)

        return frames

    def test_removes_stationary_dust_and_shading(self):
        frames = self._make_synthetic_stack(num_frames=8)
        corrected = SimpleStitcher.correct_sensor_artifacts(frames)

        self.assertEqual(len(corrected), len(frames))
        for orig, corr in zip(frames, corrected):
            self.assertEqual(orig.shape, corr.shape)
            self.assertEqual(corr.dtype, np.uint8)

        # Compare stationary background uniformity before vs after
        orig_bg = np.percentile(np.stack(frames, axis=0), 85, axis=0).mean(axis=2)
        corr_bg = np.percentile(np.stack(corrected, axis=0), 85, axis=0).mean(axis=2)

        # Standard deviation of background should drop significantly
        self.assertLess(np.std(corr_bg), np.std(orig_bg) * 0.5)

        # Opaque dust spot at (x=220, y=140) should no longer be dark in the background estimate
        orig_dust_val = orig_bg[140, 220]
        corr_dust_val = corr_bg[140, 220]
        self.assertLess(orig_dust_val, 80.0)
        self.assertGreater(corr_dust_val, 160.0)

    def test_skips_when_too_few_frames_or_mismatched_shapes(self):
        frames = self._make_synthetic_stack(num_frames=3)
        out_few = SimpleStitcher.correct_sensor_artifacts(frames, min_frames=4)
        self.assertEqual(len(out_few), 3)
        for a, b in zip(frames, out_few):
            self.assertTrue(np.array_equal(a, b))

        frames4 = self._make_synthetic_stack(num_frames=4)
        frames4[2] = frames4[2][:100, :100]
        out_mismatch = SimpleStitcher.correct_sensor_artifacts(frames4, min_frames=4)
        self.assertEqual(out_mismatch[2].shape, (100, 100, 3))

    def test_stitcher_with_sensor_correction(self):
        frames = self._make_synthetic_stack(num_frames=6)
        stitcher = SimpleStitcher(
            detector='sift',
            min_inliers=8,
            crop=True,
            exposure_compensation=True,
            equalize=False,
            sensor_correction=True,
        )
        pano = stitcher.stitch(frames, img_loader=lambda x: x)
        self.assertIsNotNone(pano)
        self.assertGreater(pano.shape[1], frames[0].shape[1])


if __name__ == '__main__':
    unittest.main()
