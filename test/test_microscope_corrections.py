import os
import unittest
import numpy as np
import cv2
import sys

sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('../src'))

from stitch import (
    apply_background_white_balance,
    apply_defringe,
    apply_radial_vignette,
    apply_local_contrast,
    apply_denoise,
    correct_illumination_poly,
    correct_illumination_rgb_div
)


class TestMicroscopeCorrections(unittest.TestCase):

    def setUp(self):
        # Resolve test output directory with suffix in testdata
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.out_dir = os.path.join(base_dir, 'testdata', 'result_microscope')
        os.makedirs(self.out_dir, exist_ok=True)

        # Create a synthetic microscope test image with background, specimen, and black border
        self.h, self.w = 300, 400
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Valid center region with yellowish background (mimicking halogen lamp)
        self.valid_y, self.valid_x = slice(20, 280), slice(20, 380)
        self.img[self.valid_y, self.valid_x] = [180, 220, 240]  # BGR: warm yellow-white

        # Dark specimen in the middle
        cv2.circle(self.img, (200, 150), 40, (50, 80, 40), -1)

    def test_background_white_balance(self):
        wb = apply_background_white_balance(self.img, target_val=250.0)
        self.assertEqual(wb.shape, self.img.shape)
        # Background pixels should have roughly equal B, G, R
        sample_bg = wb[50, 50]
        self.assertTrue(abs(int(sample_bg[0]) - int(sample_bg[1])) < 15)
        self.assertTrue(abs(int(sample_bg[1]) - int(sample_bg[2])) < 15)
        # Black border should remain zero
        self.assertTrue(np.all(wb[5, 5] == [0, 0, 0]))

        # Save result image
        cv2.imwrite(os.path.join(self.out_dir, '01_background_white_balance.jpg'), wb)

    def test_defringe(self):
        # Add high-saturation fringe on edge
        fringe_img = self.img.copy()
        cv2.circle(fringe_img, (200, 150), 42, (255, 0, 128), 2)  # purple fringe
        defringed = apply_defringe(fringe_img)
        self.assertEqual(defringed.shape, fringe_img.shape)
        self.assertTrue(np.all(defringed[5, 5] == [0, 0, 0]))

        # Save result image
        cv2.imwrite(os.path.join(self.out_dir, '02_defringe.jpg'), defringed)

    def test_radial_vignette(self):
        vignetted = apply_radial_vignette(self.img, strength=0.25)
        self.assertEqual(vignetted.shape, self.img.shape)
        # Outer valid corner should be brighter after vignette boost
        self.assertGreaterEqual(int(vignetted[25, 25, 1]), int(self.img[25, 25, 1]))

        # Save result image
        cv2.imwrite(os.path.join(self.out_dir, '04_radial_vignette.jpg'), vignetted)

    def test_local_contrast(self):
        lc = apply_local_contrast(self.img)
        self.assertEqual(lc.shape, self.img.shape)
        self.assertTrue(np.all(lc[5, 5] == [0, 0, 0]))

        # Save result image
        cv2.imwrite(os.path.join(self.out_dir, '05_local_contrast.jpg'), lc)

    def test_denoise(self):
        noisy = np.clip(self.img.astype(np.int16) + np.random.randint(-15, 15, self.img.shape), 0, 255).astype(np.uint8)
        noisy[:20, :] = 0
        noisy[-20:, :] = 0
        denoised = apply_denoise(noisy)
        self.assertEqual(denoised.shape, self.img.shape)
        self.assertTrue(np.all(denoised[5, 5] == [0, 0, 0]))

        # Save result images
        cv2.imwrite(os.path.join(self.out_dir, '06_noisy.jpg'), noisy)
        cv2.imwrite(os.path.join(self.out_dir, '06_denoised.jpg'), denoised)

    def test_pipeline_integration(self):
        import stitch
        ini_content = f"""[OPTIONS]
img_dir = ../testdata/img1
out_dir = {self.out_dir}
final_megapix = 2
confidence_threshold = 0.5
output = output
fixborder = True
correct_gradient = True
bg_white_balance = True
defringe = True
vignette_correction = True
local_contrast = True
denoise = True
"""
        with open('stitch.ini', 'w') as f:
            f.write(ini_content)

        stitch.main()

        # Verify all pipeline output files are created in result_microscope
        expected_files = [
            'output.jpg',
            'outputfixed.jpg',
            'output_rgb_division.jpg',
            'output_poly2d.jpg',
            'output_poly3d.jpg',
            'out_rgbdiv_fixed.jpg',
            'out_poly2d_fixed.jpg',
            'out_poly3d_fixed.jpg'
        ]
        for fname in expected_files:
            fpath = os.path.join(self.out_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Expected file {fname} not found in {self.out_dir}")


if __name__ == '__main__':
    unittest.main()
