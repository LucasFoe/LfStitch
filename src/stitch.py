# Copyright (c) 2025 Lucas Foerderer
# SPDX-License-Identifier: MIT

import os
import re
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

def natural_sort_key(s):
    """Sort key for natural / alphanumerical order (e.g. file9 before file10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def calculate_mean_color(output, border, black_thresh=15):
    """
    Robustly estimate a fill color for unmapped pixels near a given border.
    Samples the actual specimen background on that side of the image,
    skipping dark unmapped regions and multi-band blending transition ramps.
    """
    height, width, _ = output.shape
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    m1 = max(1, width // 3)
    m2 = max(1, height // 3)

    if border == 'left':
        reg_gray = gray[:, :m1]
        reg_bgr = output[:, :m1]
    elif border == 'right':
        reg_gray = gray[:, -m1:]
        reg_bgr = output[:, -m1:]
    elif border == 'top':
        reg_gray = gray[:m2, :]
        reg_bgr = output[:m2, :]
    elif border == 'bottom':
        reg_gray = gray[-m2:, :]
        reg_bgr = output[-m2:, :]
    else:
        reg_gray = gray
        reg_bgr = output

    # Reject unmapped pixels and dark transition ramps (gray <= 40)
    valid = reg_gray > 40
    if not np.any(valid):
        valid = reg_gray > int(black_thresh)
    if not np.any(valid):
        return np.array([245, 245, 245], dtype=int)

    # In brightfield microscopy, the background corresponds to upper luminance percentiles
    valid_gray = reg_gray[valid]
    p75 = np.percentile(valid_gray, 75)
    bg_mask = valid & (reg_gray >= p75)
    bg_pixels = reg_bgr[bg_mask]
    if len(bg_pixels) == 0:
        bg_pixels = reg_bgr[valid]

    median_color = np.median(bg_pixels, axis=0)
    return np.clip(median_color, 0, 255).astype(int)


def get_valid_mask(img_bgr, black_thresh=0):
    """Mask representing valid pixels (excluding unmapped black border)."""
    return ~np.all(img_bgr <= int(black_thresh), axis=-1)


def correct_illumination_poly(img_bgr, degree=2, target_bg=245.0, black_thresh=8):
    """
    Fit a 2D polynomial surface to background samples (excluding black borders)
    and divide the image by the fitted illumination model in LAB color space.
    """
    h, w = img_bgr.shape[:2]
    valid_mask = get_valid_mask(img_bgr, black_thresh)
    if np.sum(valid_mask) < 100:
        return img_bgr.copy()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    step = max(5, int(min(h, w) / 80))
    grid_y, grid_x = np.mgrid[0:h:step, 0:w:step]
    y_flat = grid_y.ravel().astype(np.float32) / h
    x_flat = grid_x.ravel().astype(np.float32) / w
    l_samples = L[grid_y, grid_x].ravel().astype(np.float32)
    valid_samples = valid_mask[grid_y, grid_x].ravel()

    # Identify background samples within valid area using local dilation
    l_valid = L.copy()
    l_valid[~valid_mask] = 0
    k = max(15, (int(min(h, w) * 0.05) | 1))
    l_max = cv2.dilate(l_valid, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg_mask = (L >= (l_max - 5)) & valid_mask
    bg_sample_mask = bg_mask[grid_y, grid_x].ravel() & valid_samples

    if np.sum(bg_sample_mask) < 20:
        bg_sample_mask = valid_samples

    x_bg = x_flat[bg_sample_mask]
    y_bg = y_flat[bg_sample_mask]
    l_bg = l_samples[bg_sample_mask]

    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            terms.append((x_bg ** (d - i)) * (y_bg ** i))
    M = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(M, l_bg, rcond=None)

    y_full, x_full = np.mgrid[0:h, 0:w]
    x_norm = x_full.astype(np.float32) / w
    y_norm = y_full.astype(np.float32) / h

    bg_model = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for d in range(degree + 1):
        for i in range(d + 1):
            bg_model += coeffs[idx] * (x_norm ** (d - i)) * (y_norm ** i)
            idx += 1

    bg_model = np.clip(bg_model, 10.0, 255.0)
    L_corr = np.zeros_like(L, dtype=np.float32)
    L_corr[valid_mask] = np.clip((L[valid_mask].astype(np.float32) / bg_model[valid_mask]) * target_bg, 0, 255)

    lab_corr = cv2.merge([L_corr.astype(np.uint8), A, B])
    res = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)
    res[~valid_mask] = 0
    return res


def correct_illumination_rgb_div(img_bgr, target_bg=245.0, black_thresh=8):
    """
    Estimate illumination background via morphological dilation per RGB channel
    (pre-filling black borders to prevent boundary bleed) and apply division normalization.
    """
    h, w = img_bgr.shape[:2]
    valid_mask = get_valid_mask(img_bgr, black_thresh)
    if np.sum(valid_mask) < 100:
        return img_bgr.copy()

    # Pre-fill black region with median valid color to prevent edge bleed during dilation/blur
    median_color = np.median(img_bgr[valid_mask], axis=0)
    filled = img_bgr.copy()
    filled[~valid_mask] = median_color

    max_dim = max(h, w)
    scale = min(1.0, 500.0 / max_dim)
    ksize = int(max(15, min(h, w) * 0.08 * scale)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    out_channels = []
    for ch_idx in range(3):
        ch = filled[:, :, ch_idx]
        ch_small = cv2.resize(ch, (0, 0), fx=scale, fy=scale)
        bg_ch = cv2.morphologyEx(ch_small, cv2.MORPH_DILATE, kernel)
        bg_ch = cv2.GaussianBlur(bg_ch, (0, 0), sigmaX=30 * scale, sigmaY=30 * scale)
        bg_full = cv2.resize(bg_ch, (w, h)).astype(np.float32)
        bg_full = np.clip(bg_full, 10.0, 255.0)

        orig_ch = img_bgr[:, :, ch_idx].astype(np.float32)
        corr_ch = np.zeros((h, w), dtype=np.uint8)
        corr_ch[valid_mask] = np.clip((orig_ch[valid_mask] / bg_full[valid_mask]) * target_bg, 0, 255).astype(np.uint8)
        out_channels.append(corr_ch)

    res = cv2.merge(out_channels)
    res[~valid_mask] = 0
    return res


def apply_background_white_balance(img_bgr, target_val=250.0, black_thresh=8):
    """
    1. Background White Balance:
    Identifies background pixels in valid areas and normalizes RGB gains
    so that the background becomes neutral (white/gray), eliminating lamp color casts.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid_mask = gray > black_thresh
    if np.sum(valid_mask) < 100:
        return img_bgr.copy()

    # Find brightest 10% valid pixels to represent background
    l_vals = gray[valid_mask]
    thresh = np.percentile(l_vals, 90)
    bg_mask = valid_mask & (gray >= thresh)
    if np.sum(bg_mask) < 20:
        bg_mask = valid_mask

    bg_color = np.median(img_bgr[bg_mask], axis=0).astype(np.float32)  # B, G, R
    bg_color = np.maximum(bg_color, 1.0)
    gains = target_val / bg_color
    max_gain = np.max(gains)
    if max_gain > 2.0:
        gains = gains * (2.0 / max_gain)

    out = img_bgr.astype(np.float32).copy()
    for c in range(3):
        out[valid_mask, c] = np.clip(out[valid_mask, c] * gains[c], 0, 255)
    out[~valid_mask] = 0
    return out.astype(np.uint8)


def apply_defringe(img_bgr, threshold=20, radius=3, black_thresh=8):
    """
    2. Chromatic Aberration Correction / Defringing:
    Detects high-contrast edge zones and suppresses chromatic color fringing (e.g. purple/green edges).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid_mask = gray > black_thresh
    if np.sum(valid_mask) < 100:
        return img_bgr.copy()

    edges = cv2.Canny(gray, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    edge_zone = cv2.dilate(edges, kernel) > 0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    fringe_candidates = edge_zone & valid_mask & (s_ch > threshold)
    s_corr = s_ch.copy()
    s_corr[fringe_candidates] = (s_corr[fringe_candidates] * 0.25).astype(np.uint8)

    hsv_corr = cv2.merge([h_ch, s_corr, v_ch])
    out = cv2.cvtColor(hsv_corr, cv2.COLOR_HSV2BGR)
    out[~valid_mask] = 0
    return out


def apply_radial_vignette(img_bgr, strength=0.25):
    """
    4. Radial Vignetting Correction:
    Compensates for optical lens light fall-off from center to edges on individual frames.
    """
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.ogrid[:h, :w]
    max_r2 = cx**2 + cy**2
    r2_norm = ((X - cx)**2 + (Y - cy)**2) / max_r2
    vignette = 1.0 - strength * r2_norm
    vignette = np.clip(vignette, 0.1, 1.0)
    return np.clip(img_bgr.astype(np.float32) / vignette[:, :, None], 0, 255).astype(np.uint8)


def apply_local_contrast(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8), black_thresh=8):
    """
    5. Local Contrast Enhancement (CLAHE) / Dehazing:
    Enhances fine specimen structures and textures in LAB color space without color distortion.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid_mask = gray > black_thresh
    if np.sum(valid_mask) < 100:
        return img_bgr.copy()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_clahe = clahe.apply(L)
    L_out = np.zeros_like(L)
    L_out[valid_mask] = L_clahe[valid_mask]
    out = cv2.cvtColor(cv2.merge([L_out, A, B]), cv2.COLOR_LAB2BGR)
    out[~valid_mask] = 0
    return out


def apply_denoise(img_bgr, d=5, sigma_color=25, sigma_space=25, black_thresh=8):
    """
    6. Microscope Denoising:
    Applies edge-preserving bilateral filtering to suppress sensor noise while preserving sharp cell walls.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid_mask = gray > black_thresh
    if np.sum(valid_mask) < 100:
        return img_bgr.copy()

    denoised = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    denoised[~valid_mask] = 0
    return denoised


def scale_image(img, final_megapix):
    if img is None or final_megapix <= 0:
        return img
    h, w = img.shape[:2]
    current_megapix = (w * h) / 1e6
    if current_megapix > final_megapix:
        scale = (final_megapix / current_megapix) ** 0.5
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def apply_post_enhancements(img, denoise=False, defringe=False, bg_white_balance=False, local_contrast=False):
    if img is None:
        return None
    res = img
    if denoise:
        res = apply_denoise(res)
    if defringe:
        res = apply_defringe(res)
    if bg_white_balance:
        res = apply_background_white_balance(res)
    if local_contrast:
        res = apply_local_contrast(res)
    return res


# python
def fix_border(output, coverage_mask=None, extend=5):
    """
    Seamlessly fill unmapped border canvas regions outside the stitched panorama.
    Guarantees that interior pixels (e.g. dark spores or zero pixels inside specimens)
    are never touched or overwritten.
    Each unmapped border component is filled using the median color of the valid
    specimen background directly adjacent to that specific component.
    """
    if output is None:
        return None

    height, width = output.shape[:2]
    outputfixed = output.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    black = np.all(output == 0, axis=-1).astype(np.uint8)

    # Use coverage_mask if provided and valid; otherwise use pure-black mask
    if coverage_mask is not None and coverage_mask.shape[:2] == (height, width):
        raw_unmapped = (coverage_mask == 0).astype(np.uint8)
    else:
        raw_unmapped = black

    if not raw_unmapped.any():
        return outputfixed

    # Find connected components of unmapped areas
    num_labels, labels = cv2.connectedComponents(raw_unmapped)

    # Only components that touch the canvas perimeter are true unmapped borders
    border_labels = set()
    for x in range(width):
        if raw_unmapped[0, x]: border_labels.add(labels[0, x])
        if raw_unmapped[height - 1, x]: border_labels.add(labels[height - 1, x])
    for y in range(height):
        if raw_unmapped[y, 0]: border_labels.add(labels[y, 0])
        if raw_unmapped[y, width - 1]: border_labels.add(labels[y, width - 1])
    border_labels.discard(0)

    if not border_labels:
        return outputfixed

    unmapped_mask = np.isin(labels, list(border_labels))

    for lbl in border_labels:
        comp_mask = (labels == lbl)

        # Sample adjacent valid image pixels directly bordering this component
        dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=3)
        adj_mask = (dilated > 0) & (~unmapped_mask)
        adj_pixels = output[adj_mask]

        if len(adj_pixels) == 0:
            continue

        adj_gray = gray[adj_mask]
        # Skip dark multi-band pyramid transition ramp right at the zero border
        valid = adj_pixels[adj_gray > 30]
        if len(valid) == 0:
            valid = adj_pixels

        fill_color = np.median(valid, axis=0).astype(np.uint8)

        # Fill the unmapped component
        outputfixed[comp_mask] = fill_color

        # Cover any dark boundary fade ramp right next to this component
        if extend > 0:
            seam_dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((2 * extend + 1, 2 * extend + 1), np.uint8)) > 0
            seam_mask = seam_dilated & (~comp_mask) & (gray < 40)
            if seam_mask.any():
                outputfixed[seam_mask] = fill_color

    return outputfixed


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
        'detector': 'sift',
        'correct_gradient': True,
        'bg_white_balance': False,
        'defringe': False,
        'vignette_correction': False,
        'local_contrast': False,
        'denoise': False,
        'sensor_correction': False
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
    correct_gradient = config.getboolean('OPTIONS', 'correct_gradient', fallback=default_options['correct_gradient'])
    bg_white_balance = config.getboolean('OPTIONS', 'bg_white_balance', fallback=default_options['bg_white_balance'])
    defringe = config.getboolean('OPTIONS', 'defringe', fallback=default_options['defringe'])
    vignette_correction = config.getboolean('OPTIONS', 'vignette_correction', fallback=default_options['vignette_correction'])
    local_contrast = config.getboolean('OPTIONS', 'local_contrast', fallback=default_options['local_contrast'])
    denoise = config.getboolean('OPTIONS', 'denoise', fallback=default_options['denoise'])
    sensor_correction = config.getboolean('OPTIONS', 'sensor_correction', fallback=default_options['sensor_correction'])

    # Log the option values
    logger.info("Option img_dir: %s", img_dir)
    logger.info("Option out_dir: %s", out_dir)
    logger.info("Option final_megapix: %f", final_megapix)
    logger.info("Option try_use_gpu: %s", try_use_gpu)
    logger.info("Option confidence_threshold: %f", confidence_threshold)
    logger.info("Option output: %s", outfile)
    logger.info("Option fixborder: %s", fixborder)
    logger.info("Option detector: %s", detector_name)
    logger.info("Option correct_gradient: %s", correct_gradient)
    logger.info("Option bg_white_balance: %s", bg_white_balance)
    logger.info("Option defringe: %s", defringe)
    logger.info("Option vignette_correction: %s", vignette_correction)
    logger.info("Option local_contrast: %s", local_contrast)
    logger.info("Option denoise: %s", denoise)
    logger.info("Option sensor_correction: %s", sensor_correction)

    logger.info("Input Directory: %s", os.path.abspath(img_dir))
    filelist = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir), key=natural_sort_key)]
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
        num_bands=5,
        sensor_correction=sensor_correction
    )
    # Pre-stitching corrections (Radial vignette correction)
    if vignette_correction:
        imgs = [cv2.imread(f, cv2.IMREAD_COLOR) for f in filelist]
        if any(im is None for im in imgs):
            raise ValueError("One or more images failed to load from filelist.")

        logger.info("Applying pre-stitch radial vignette correction on input tiles...")
        imgs = [apply_radial_vignette(im) for im in imgs]

        output = stitcher.stitch(imgs, img_loader=lambda p: p if isinstance(p, np.ndarray) else cv2.imread(p, cv2.IMREAD_COLOR))
    else:
        output = stitcher.stitch(filelist)

    if output is None:
        logger.error("Stitcher returned no output (None). Exiting.")
        raise RuntimeError("Stitching failed, no output image produced")

    # 1. Border fix applied directly after merge while unmapped areas are strictly (0, 0, 0)
    cov_mask = getattr(stitcher, 'last_coverage_mask', None)
    output_fixed = None
    if fixborder:
        try:
            logger.info("Applying border fix directly after merge on raw stitched output...")
            output_fixed = fix_border(output, coverage_mask=cov_mask)
        except Exception:
            logger.exception("Failed to fix border directly after merge, falling back to original")
            output_fixed = output.copy()

    # 2. Gradient correction & border-fixed gradient variants (evaluated while black borders are strictly preserved)
    rgb_div = None
    poly2d = None
    poly3d = None
    rgb_div_fixed = None
    poly2d_fixed = None
    poly3d_fixed = None

    if correct_gradient:
        try:
            logger.info("Performing illumination gradient corrections...")
            rgb_div = correct_illumination_rgb_div(output)
            poly2d = correct_illumination_poly(output, degree=2)
            poly3d = correct_illumination_poly(output, degree=3)

            if fixborder:
                rgb_div_fixed = fix_border(rgb_div, coverage_mask=cov_mask)
                poly2d_fixed = fix_border(poly2d, coverage_mask=cov_mask)
                poly3d_fixed = fix_border(poly3d, coverage_mask=cov_mask)
        except Exception:
            logger.exception("Failed during illumination gradient correction")

    # 3. Optional post-stitching microscope enhancements (Denoise, Defringe, White balance, Local contrast)
    if any([denoise, defringe, bg_white_balance, local_contrast]):
        logger.info("Applying post-stitch enhancements...")
        output = apply_post_enhancements(output, denoise, defringe, bg_white_balance, local_contrast)
        if output_fixed is not None:
            output_fixed = apply_post_enhancements(output_fixed, denoise, defringe, bg_white_balance, local_contrast)
        if rgb_div is not None:
            rgb_div = apply_post_enhancements(rgb_div, denoise, defringe, bg_white_balance, local_contrast)
        if poly2d is not None:
            poly2d = apply_post_enhancements(poly2d, denoise, defringe, bg_white_balance, local_contrast)
        if poly3d is not None:
            poly3d = apply_post_enhancements(poly3d, denoise, defringe, bg_white_balance, local_contrast)
        if rgb_div_fixed is not None:
            rgb_div_fixed = apply_post_enhancements(rgb_div_fixed, denoise, defringe, bg_white_balance, local_contrast)
        if poly2d_fixed is not None:
            poly2d_fixed = apply_post_enhancements(poly2d_fixed, denoise, defringe, bg_white_balance, local_contrast)
        if poly3d_fixed is not None:
            poly3d_fixed = apply_post_enhancements(poly3d_fixed, denoise, defringe, bg_white_balance, local_contrast)

    # 4. Resize output images if final_megapix is specified (> 0)
    if final_megapix > 0:
        h, w = output.shape[:2]
        current_megapix = (w * h) / 1e6
        if current_megapix > final_megapix:
            logger.info("Scaling final images from %.2f MP (%dx%d) to %.2f MP target",
                        current_megapix, w, h, final_megapix)
            output = scale_image(output, final_megapix)
            if output_fixed is not None:
                output_fixed = scale_image(output_fixed, final_megapix)
            if rgb_div is not None:
                rgb_div = scale_image(rgb_div, final_megapix)
            if poly2d is not None:
                poly2d = scale_image(poly2d, final_megapix)
            if poly3d is not None:
                poly3d = scale_image(poly3d, final_megapix)
            if rgb_div_fixed is not None:
                rgb_div_fixed = scale_image(rgb_div_fixed, final_megapix)
            if poly2d_fixed is not None:
                poly2d_fixed = scale_image(poly2d_fixed, final_megapix)
            if poly3d_fixed is not None:
                poly3d_fixed = scale_image(poly3d_fixed, final_megapix)

    # 5. Output file paths and writing
    os.makedirs(out_dir, exist_ok=True)

    base_name = outfile
    out_path = os.path.abspath(os.path.join(out_dir, f'{base_name}.jpg'))
    fixed_out_path = os.path.abspath(os.path.join(out_dir, f'{base_name}fixed.jpg'))

    # Write stitched output
    cv2.imwrite(out_path, output)
    logger.info("Output file: %s", out_path)

    # Fixed output
    if fixborder and output_fixed is not None:
        cv2.imwrite(fixed_out_path, output_fixed)
        logger.info("Fixed output file: %s", fixed_out_path)
    else:
        if os.path.exists(fixed_out_path):
            try:
                os.remove(fixed_out_path)
            except OSError:
                pass

    pfx = 'out' if base_name == 'output' else base_name
    rgb_div_path = os.path.abspath(os.path.join(out_dir, f'{base_name}_rgb_division.jpg'))
    poly2d_path = os.path.abspath(os.path.join(out_dir, f'{base_name}_poly2d.jpg'))
    poly3d_path = os.path.abspath(os.path.join(out_dir, f'{base_name}_poly3d.jpg'))
    rgb_div_fixed_path = os.path.abspath(os.path.join(out_dir, f'{pfx}_rgbdiv_fixed.jpg'))
    poly2d_fixed_path = os.path.abspath(os.path.join(out_dir, f'{pfx}_poly2d_fixed.jpg'))
    poly3d_fixed_path = os.path.abspath(os.path.join(out_dir, f'{pfx}_poly3d_fixed.jpg'))

    gradient_files = [rgb_div_path, poly2d_path, poly3d_path]
    fixed_gradient_files = [rgb_div_fixed_path, poly2d_fixed_path, poly3d_fixed_path]

    if correct_gradient:
        if rgb_div is not None:
            cv2.imwrite(rgb_div_path, rgb_div)
            logger.info("Gradient-corrected output (RGB division): %s", rgb_div_path)
        if poly2d is not None:
            cv2.imwrite(poly2d_path, poly2d)
            logger.info("Gradient-corrected output (Poly 2D): %s", poly2d_path)
        if poly3d is not None:
            cv2.imwrite(poly3d_path, poly3d)
            logger.info("Gradient-corrected output (Poly 3D): %s", poly3d_path)

        if fixborder:
            if rgb_div_fixed is not None:
                cv2.imwrite(rgb_div_fixed_path, rgb_div_fixed)
                logger.info("Fixed gradient output (RGB division): %s", rgb_div_fixed_path)
            if poly2d_fixed is not None:
                cv2.imwrite(poly2d_fixed_path, poly2d_fixed)
                logger.info("Fixed gradient output (Poly 2D): %s", poly2d_fixed_path)
            if poly3d_fixed is not None:
                cv2.imwrite(poly3d_fixed_path, poly3d_fixed)
                logger.info("Fixed gradient output (Poly 3D): %s", poly3d_fixed_path)
        else:
            for fpath in fixed_gradient_files:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
    else:
        for fpath in gradient_files + fixed_gradient_files:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass


if __name__ == '__main__':
    main()
