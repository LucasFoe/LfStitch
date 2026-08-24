import os
import unittest

import cv2
import numpy as np


class TestStitchingProgram(unittest.TestCase):

    def setUp(self):
        # Set up test directories and files
        self.test_dir = '../testdata'
        self.img_dir = os.path.join(self.test_dir, 'img1')
        self.out_dir = os.path.join(self.test_dir, 'result1')
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create a dummy configuration file
        with open('stitch.ini', 'w') as config_file:
            config_file.write("[OPTIONS]\n")
            config_file.write(f"img_dir = {self.img_dir}\n")
            config_file.write(f"out_dir = {self.out_dir}\n")
            config_file.write("final_megapix = 5\n")
            config_file.write("try_use_gpu = False\n")
            config_file.write("confidence_threshold = 0.5\n")
            config_file.write("output = output\n")
            config_file.write("fixborder = True\n")

    def tearDown(self):
        pass

    def test_image_stitching(self):
        # Import the main script and execute
        import stitch
        stitch.main()
        # Check if the output files are created
        stitched_file = os.path.join(self.out_dir, 'output.jpg')
        fixed_file = os.path.join(self.out_dir, 'outputfixed.jpg')


        self.assertTrue(os.path.exists(stitched_file))
        self.assertTrue(os.path.exists(fixed_file))

        # Load the images and check their properties
        stitched_img = cv2.imread(stitched_file)
        fixed_img = cv2.imread(fixed_file)

        self.assertIsNotNone(stitched_img)
        self.assertIsNotNone(fixed_img)

        self.assertEqual(stitched_img.shape, fixed_img.shape)

        # Check if the fixed image has no black borders
        self.assertFalse(np.any(np.all(fixed_img == [0, 0, 0], axis=-1)))


if __name__ == '__main__':
    unittest.main()