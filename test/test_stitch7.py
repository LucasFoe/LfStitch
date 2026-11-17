import unittest
import os
import cv2
import numpy as np


class TestStitchingProgram7(unittest.TestCase):

    def setUp(self):
        self.test_dir = '../testdata'
        self.img_dir = os.path.join(self.test_dir, 'img7')
        self.out_dir = os.path.join(self.test_dir, 'result7')
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def tearDown(self):
        pass

    def test_image_stitching_and_border_fix(self):
        # Configure with fixborder=True, correct_gradient=True
        with open('stitch.ini', 'w') as config_file:
            config_file.write("[OPTIONS]\n")
            config_file.write(f"img_dir = {self.img_dir}\n")
            config_file.write(f"out_dir = {self.out_dir}\n")
            config_file.write("final_megapix = 5\n")
            config_file.write("confidence_threshold = 0.1\n")
            config_file.write("output = output\n")
            config_file.write("fixborder = True\n")
            config_file.write("correct_gradient = True\n")

        import stitch
        stitch.main()

        stitched_file = os.path.join(self.out_dir, 'output.jpg')
        fixed_file = os.path.join(self.out_dir, 'outputfixed.jpg')

        self.assertTrue(os.path.exists(stitched_file))
        self.assertTrue(os.path.exists(fixed_file))

        fixed_img = cv2.imread(fixed_file)
        self.assertIsNotNone(fixed_img)

        # Check corners of fixed image: they should match local borders (not black or dark fade ramp)
        gray_fixed = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)
        self.assertGreater(gray_fixed[10, 10], 100, "Top-left corner not properly filled")
        self.assertGreater(gray_fixed[10, -10], 150, "Top-right corner not properly filled")
        self.assertGreater(gray_fixed[-10, 10], 150, "Bottom-left corner not properly filled")
        self.assertGreater(gray_fixed[-10, -10], 150, "Bottom-right corner not properly filled")

        # Verify that pure black pixels inside the specimen are preserved (not overwritten by border fill)
        self.assertGreater(np.all(fixed_img == 0, axis=-1).sum(), 0, "Interior zero pixels should be preserved")

        # Verify gradient and fixed gradient files exist
        for fname in ['output_rgb_division.jpg', 'output_poly2d.jpg', 'output_poly3d.jpg',
                      'out_rgbdiv_fixed.jpg', 'out_poly2d_fixed.jpg', 'out_poly3d_fixed.jpg']:
            fpath = os.path.join(self.out_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"File {fname} should exist when flags are True")

    def test_flags_false_no_files_created(self):
        # Configure with fixborder=False, correct_gradient=False
        with open('stitch.ini', 'w') as config_file:
            config_file.write("[OPTIONS]\n")
            config_file.write(f"img_dir = {self.img_dir}\n")
            config_file.write(f"out_dir = {self.out_dir}\n")
            config_file.write("final_megapix = 5\n")
            config_file.write("confidence_threshold = 0.1\n")
            config_file.write("output = output\n")
            config_file.write("fixborder = False\n")
            config_file.write("correct_gradient = False\n")

        import stitch
        stitch.main()

        # output.jpg should exist
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'output.jpg')))

        # None of the fixed or gradient files should exist
        forbidden_files = [
            'outputfixed.jpg',
            'output_rgb_division.jpg',
            'output_poly2d.jpg',
            'output_poly3d.jpg',
            'out_rgbdiv_fixed.jpg',
            'out_poly2d_fixed.jpg',
            'out_poly3d_fixed.jpg'
        ]
        for fname in forbidden_files:
            fpath = os.path.join(self.out_dir, fname)
            self.assertFalse(os.path.exists(fpath), f"File {fname} should NOT exist when flags are False")


if __name__ == '__main__':
    unittest.main()
