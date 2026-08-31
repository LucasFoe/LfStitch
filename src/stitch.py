# Copyright (c) 2025 Lucas Foerderer
# SPDX-License-Identifier: MIT

import os
import numpy as np
import cv2
import logging
import configparser
from lfstitcher import SimpleStitcher

# Log file path
log_file_path = 'stitch.log'

# Check if log file exists and its size
if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 10 * 1024 * 1024:  # 10MB
    os.remove(log_file_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

def calculate_mean_color(output, border, black_thresh=8, ring=10, lum_clip=(5, 95)):
    """
    Robustly estimate a fill color for unmapped pixels near a given border.

    - black_thresh: gray threshold to consider pixels as unmapped (<= threshold).
    - ring: number of pixels to look around the unmapped area (dilation radius).
    - lum_clip: percentile range for luminance-based outlier rejection.
    """
    height, width, _ = output.shape
    img_f = output.astype(np.float32)

    # Unmapped mask (treat near-black as unmapped)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    black_mask = gray <= int(black_thresh)

    # Build a ring around unmapped pixels: dilate then subtract the original unmapped area
    kernel = np.ones((2 * int(ring) + 1, 2 * int(ring) + 1), dtype=np.uint8)
    dilated = cv2.dilate(black_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring_mask = dilated & (~black_mask)

    # Restrict to the same side regions used by fix_border (left/right third, top/bottom third)
    m1 = width // 3
    m2 = height // 3
    region = np.zeros((height, width), dtype=bool)
    if border == 'left':
        region[:, :m1] = True
    elif border == 'right':
        region[:, -m1:] = True
    elif border == 'top':
        region[:m2, :] = True
    elif border == 'bottom':
        region[-m2:, :] = True
    else:
        region[:] = True  # fallback: whole image

    candidates = ring_mask & region

    # Fallbacks if the ring is empty in that region
    if not np.any(candidates):
        candidates = (~black_mask) & region
    if not np.any(candidates):
        candidates = ~black_mask
    if not np.any(candidates):
        return np.array([0, 0, 0], dtype=int)

    # Luminance-based outlier rejection (percentile clipping)
    lum = img_f[..., 0] * 0.114 + img_f[..., 1] * 0.587 + img_f[..., 2] * 0.299
    lum_vals = lum[candidates]
    if lum_vals.size > 0:
        lo, hi = np.percentile(lum_vals, lum_clip)
        candidates = candidates & (lum >= lo) & (lum <= hi)

    # Final fallback if clipping removed everything
    if not np.any(candidates):
        candidates = (~black_mask)

    colors = img_f[candidates]
    if colors.size == 0:
        return np.array([0, 0, 0], dtype=int)

    # Robust estimate: per-channel median
    median_color = np.median(colors, axis=0)
    return np.clip(median_color, 0, 255).astype(int)

# python
def fix_border(output):
    if output is None:
        return None

    height, width = output.shape[:2]
    outputfixed = output.copy().astype(np.int32)

    # Compute per-border mean colors (returns int array)
    mean_color_left = calculate_mean_color(output, 'left').astype(np.int32)
    mean_color_right = calculate_mean_color(output, 'right').astype(np.int32)
    mean_color_top = calculate_mean_color(output, 'top').astype(np.int32)
    mean_color_bottom = calculate_mean_color(output, 'bottom').astype(np.int32)

    m1 = max(1, width // 3)
    m2 = max(1, height // 3)

    # Build dilated black mask
    extend = 10
    black = np.all(output == [0, 0, 0], axis=-1).astype(np.uint8)  # HxW 0/1
    kernel = np.ones((2 * extend + 1, 2 * extend + 1), dtype=np.uint8)
    black_dilated = cv2.dilate(black, kernel, iterations=1).astype(bool)  # HxW boolean

    # Border-region masks (relative coordinates)
    left_mask = black_dilated[:, :m1]
    right_mask = black_dilated[:, -m1:] if m1 > 0 else np.zeros((height, 0), dtype=bool)
    top_mask = black_dilated[:m2, :]
    bottom_mask = black_dilated[-m2:, :] if m2 > 0 else np.zeros((0, width), dtype=bool)

    # Assign mean color to masked pixels using explicit coords (avoid boolean-index broadcasting)
    if left_mask.any():
        rows, cols = np.nonzero(left_mask)
        outputfixed[rows, cols] = mean_color_left

    if right_mask.any():
        rows, cols = np.nonzero(right_mask)
        cols_full = cols + (width - m1)
        outputfixed[rows, cols_full] = mean_color_right

    if top_mask.any():
        rows, cols = np.nonzero(top_mask)
        outputfixed[rows, cols] = mean_color_top

    if bottom_mask.any():
        rows, cols = np.nonzero(bottom_mask)
        rows_full = rows + (height - m2)
        outputfixed[rows_full, cols] = mean_color_bottom

    # Overall mean color (slightly darker) and replace any remaining pure-black pixels
    mean_color = np.mean([mean_color_left, mean_color_right, mean_color_top, mean_color_bottom], axis=0).astype(np.int32)
    mean_color = np.clip(mean_color - 5, 0, 255)

    remaining_black = np.all(outputfixed == [0, 0, 0], axis=-1)
    if remaining_black.any():
        rows, cols = np.nonzero(remaining_black)
        outputfixed[rows, cols] = mean_color

    # Ensure uint8 before CLAHE and apply
    outputfixed_uint8 = np.clip(outputfixed, 0, 255).astype(np.uint8)
    return outputfixed_uint8


def main():
    config = configparser.ConfigParser()

    ini_path = os.path.abspath('stitch.ini')
    logger.info("Config file full path: %s", ini_path)
    config.read(ini_path)

    # Set default values
    default_options = {
        'img_dir': './img',
        'out_dir': './result',
        'final_megapix': 5.0,
        'try_use_gpu': True,
        'confidence_threshold': 0.5,
        'output': 'output',
        'fixborder': True,
        'detector': 'sift'
    }

    # Get values from config file or use defaults
    img_dir = config.get('OPTIONS', 'img_dir', fallback=default_options['img_dir'])
    out_dir = config.get('OPTIONS', 'out_dir', fallback=default_options['out_dir'])

    img_dir = os.path.abspath(img_dir)
    out_dir = os.path.abspath(out_dir)

    final_megapix = config.getfloat('OPTIONS', 'final_megapix', fallback=default_options['final_megapix'])
    try_use_gpu = config.getboolean('OPTIONS', 'try_use_gpu', fallback=default_options['try_use_gpu'])
    confidence_threshold = config.getfloat('OPTIONS', 'confidence_threshold', fallback=default_options['confidence_threshold'])
    outfile = config.get('OPTIONS', 'output', fallback=default_options['output'])
    fixborder = config.getboolean('OPTIONS', 'fixborder', fallback=default_options['fixborder'])
    detector_name = config.get('OPTIONS', 'detector', fallback=default_options['detector'])

    # Log the option values
    logger.info("Option img_dir: %s", img_dir)
    logger.info("Option out_dir: %s", out_dir)
    logger.info("Option final_megapix: %f", final_megapix)
    logger.info("Option try_use_gpu: %s", try_use_gpu)
    logger.info("Option confidence_threshold: %f", confidence_threshold)
    logger.info("Option output: %s", outfile)
    logger.info("Option fixborder: %s", fixborder)
    logger.info("Option detector: %s", detector_name)

    logger.info("Input Directory: %s", os.path.abspath(img_dir))
    filelist = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    logger.info("List of input files with full paths: %s", filelist)

    # map confidence_threshold -> min_inliers and instantiate SimpleStitcher with new args
    min_inliers = max(4, int(confidence_threshold * 40))
    logger.info("Mapped confidence_threshold=%f -> min_inliers=%d", confidence_threshold, min_inliers)
    stitcher = SimpleStitcher(
        detector=detector_name,
        ratio=0.8,
        ransac_thresh=3.0,
        min_inliers=min_inliers,
        warper_type="plane",
        crop=True,
        try_use_gpu=try_use_gpu,
        blender_type="multiband",
        exposure_compensation=True,
        equalize=True,
        num_bands=5
    )
    output = stitcher.stitch(filelist)

    if output is None:
        logger.error("Stitcher returned no output (None). Exiting.")
        raise RuntimeError("Stitching failed, no output image produced")

    # Resize output image if final_megapix is specified (> 0)
    if final_megapix > 0:
        h, w = output.shape[:2]
        current_megapix = (w * h) / 1e6
        if current_megapix > final_megapix:
            scale = (final_megapix / current_megapix) ** 0.5
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            logger.info("Scaling final image from %.2f MP (%dx%d) to %.2f MP (%dx%d)",
                        current_megapix, w, h, final_megapix, new_w, new_h)
            output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    base_name = outfile
    out_path = os.path.abspath(os.path.join(out_dir, f'{base_name}.jpg'))
    fixed_out_path = os.path.abspath(os.path.join(out_dir, f'{base_name}fixed.jpg'))

    # Write original output
    cv2.imwrite(out_path, output)
    logger.info("Output file: %s", out_path)

    # Conditionally perform border-fixing post-processing
    if fixborder:
        try:
            output_fixed = fix_border(output)
            cv2.imwrite(fixed_out_path, output_fixed)
            logger.info("Fixed output file: %s", fixed_out_path)
        except Exception:
            logger.exception("Failed to create fixed output, writing original as fixed")
            cv2.imwrite(fixed_out_path, output)
            logger.info("Fixed output file (fallback to original): %s", fixed_out_path)


if __name__ == '__main__':
    main()
