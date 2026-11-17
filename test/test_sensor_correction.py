import os
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from lfstitcher import SimpleStitcher


OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testdata', 'result_sensor_correction'))


class TestSensorCorrection(unittest.TestCase):
    def setUp(self):
        os.makedirs(OUT_DIR, exist_ok=True)

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

    def test_removes_stationary_dust_and_shading_with_refs(self):
        frames = self._make_synthetic_stack(num_frames=8)
        # 3 near-empty reference frames (w1, w2, w3) sharing the same stationary field
        h, w = frames[0].shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        illum = 1.0 - 0.30 * (((xx - w * 0.4) / w) ** 2 + ((yy - h * 0.5) / h) ** 2)
        smudge = 1.0 - 0.18 * np.exp(-(((xx - 90) ** 2 + (yy - 80) ** 2) / (2 * 18.0 ** 2)))
        dust = np.where(np.sqrt((xx - 220) ** 2 + (yy - 140) ** 2) <= 4.0, 0.15, 1.0).astype(np.float32)
        stationary_field = (illum * smudge * dust)[:, :, None]
        refs = [np.clip(np.full((h, w, 3), 220, dtype=np.float32) * stationary_field, 0, 255).astype(np.uint8) for _ in range(3)]
        corrected = SimpleStitcher.correct_stationary_artifacts(frames, reference_images=refs)

        self.assertEqual(len(corrected), len(frames))
        for orig, corr in zip(frames, corrected):
            self.assertEqual(orig.shape, corr.shape)
            self.assertEqual(corr.dtype, np.uint8)

        # Compare stationary background uniformity before vs after
        orig_bg = np.percentile(np.stack(frames, axis=0), 85, axis=0)
        corr_bg = np.percentile(np.stack(corrected, axis=0), 85, axis=0)

        cv2.imwrite(os.path.join(OUT_DIR, 'synthetic_frame0_before.jpg'), frames[0])
        cv2.imwrite(os.path.join(OUT_DIR, 'synthetic_frame0_after.jpg'), corrected[0])
        cv2.imwrite(os.path.join(OUT_DIR, 'synthetic_bg_before.jpg'), np.clip(orig_bg, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUT_DIR, 'synthetic_bg_after.jpg'), np.clip(corr_bg, 0, 255).astype(np.uint8))

        orig_bg_gray = orig_bg.mean(axis=2)
        corr_bg_gray = corr_bg.mean(axis=2)

        # Standard deviation of background should drop significantly
        self.assertLess(np.std(corr_bg_gray), np.std(orig_bg_gray) * 0.5)

        # Opaque dust spot at (x=220, y=140) should no longer be dark in the background estimate
        orig_dust_val = orig_bg_gray[140, 220]
        corr_dust_val = corr_bg_gray[140, 220]
        self.assertLess(orig_dust_val, 80.0)
        self.assertGreater(corr_dust_val, 160.0)

    def test_skips_when_no_refs_or_mismatched_shapes(self):
        frames = self._make_synthetic_stack(num_frames=3)
        out_none = SimpleStitcher.correct_stationary_artifacts(frames, reference_images=None)
        self.assertEqual(len(out_none), 3)
        for a, b in zip(frames, out_none):
            self.assertTrue(np.array_equal(a, b))

        out_few_refs = SimpleStitcher.correct_stationary_artifacts(frames, reference_images=frames[:1], min_frames=2)
        for a, b in zip(frames, out_few_refs):
            self.assertTrue(np.array_equal(a, b))

        bad_refs = self._make_synthetic_stack(num_frames=3)
        bad_refs[1] = bad_refs[1][:100, :100]
        out_mismatch = SimpleStitcher.correct_stationary_artifacts(frames, reference_images=bad_refs)
        for a, b in zip(frames, out_mismatch):
            self.assertTrue(np.array_equal(a, b))

    def test_stitcher_with_reference_filelist(self):
        frames = self._make_synthetic_stack(num_frames=6)
        stitcher_before = SimpleStitcher(
            detector='sift',
            min_inliers=8,
            crop=True,
            exposure_compensation=True,
            equalize=False,
        )
        pano_before = stitcher_before.stitch(frames, img_loader=lambda x: x)

        stitcher = SimpleStitcher(
            detector='sift',
            min_inliers=8,
            crop=True,
            exposure_compensation=True,
            equalize=False,
        )
        pano = stitcher.stitch(frames, img_loader=lambda x: x, reference_filelist=frames)
        self.assertIsNotNone(pano)
        self.assertGreater(pano.shape[1], frames[0].shape[1])

        if pano_before is not None:
            cv2.imwrite(os.path.join(OUT_DIR, 'synthetic_stitched_before.jpg'), pano_before)
        cv2.imwrite(os.path.join(OUT_DIR, 'synthetic_stitched_after.jpg'), pano)

    def test_real_img8_skips_without_refs(self):
        img8_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testdata', 'img8'))
        if not os.path.isdir(img8_dir):
            self.skipTest("testdata/img8 not present")
        files = sorted(f for f in os.listdir(img8_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png')))
        imgs = [cv2.imread(os.path.join(img8_dir, f), cv2.IMREAD_COLOR) for f in files]
        self.assertEqual(len(imgs), 2)

        corr = SimpleStitcher.correct_stationary_artifacts(imgs, reference_images=None)
        for a, b in zip(imgs, corr):
            self.assertTrue(np.array_equal(a, b))

    def test_reference_frames_reject_slide_particles(self):
        h, w = 200, 260
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        illum = 1.0 - 0.30 * (((xx - w * 0.4) / w) ** 2 + ((yy - h * 0.5) / h) ** 2)
        smudge = 1.0 - 0.18 * np.exp(-(((xx - 90) ** 2 + (yy - 80) ** 2) / (2 * 18.0 ** 2)))
        dust = np.where(np.sqrt((xx - 180) ** 2 + (yy - 120) ** 2) <= 4.0, 0.15, 1.0).astype(np.float32)
        stationary_field = (illum * smudge * dust)[:, :, None]

        # 3 shifted almost-empty reference frames (w1, w2, w3), each with a stray slide particle at a different position
        refs = []
        stray_positions = [(50, 50), (130, 150), (210, 60)]
        for sx, sy in stray_positions:
            slide = np.full((h, w, 3), 220, dtype=np.float32)
            cv2.circle(slide, (sx, sy), 6, (40, 40, 40), -1)
            refs.append(np.clip(slide * stationary_field, 0, 255).astype(np.uint8))

        # 2 specimen frames sharing the same stationary_field with non-overlapping specimen objects
        specimen_frames = []
        for cx in (60, 200):
            slide = np.full((h, w, 3), 220, dtype=np.float32)
            cv2.circle(slide, (cx, 100), 20, (70, 70, 70), -1)
            specimen_frames.append(np.clip(slide * stationary_field, 0, 255).astype(np.uint8))

        corrected = SimpleStitcher.correct_stationary_artifacts(specimen_frames, reference_images=refs)
        self.assertEqual(len(corrected), 2)

        orig_bg = np.max(np.stack(specimen_frames, axis=0), axis=0).mean(axis=2)
        corr_bg = np.max(np.stack(corrected, axis=0), axis=0).mean(axis=2)

        # Stationary dust at (180, 120) should be removed, and background std should drop > 50%
        self.assertGreater(corr_bg[120, 180], 150.0)
        self.assertLess(np.std(corr_bg), np.std(orig_bg) * 0.5)

        # Stray slide particles in w1/w2/w3 should be rejected by median (not amplified into bright spots)
        corrected_refs = SimpleStitcher.correct_stationary_artifacts(refs, reference_images=refs)
        ref_bg = np.median(np.stack(corrected_refs, axis=0), axis=0).mean(axis=2)
        self.assertLess(np.std(ref_bg), 10.0)


if __name__ == '__main__':
    unittest.main()
