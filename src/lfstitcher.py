import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Cache CUDA availability check so it is executed only once
def _get_cuda_device_count_cached():
    """Return cached CUDA device count to avoid repeated expensive checks."""
    if not hasattr(_get_cuda_device_count_cached, "_cached_count"):
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            try:
                _get_cuda_device_count_cached._cached_count = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception:
                _get_cuda_device_count_cached._cached_count = 0
        else:
            _get_cuda_device_count_cached._cached_count = 0
    return _get_cuda_device_count_cached._cached_count

class FeatureMatcher:
    """
    Feature matcher tuned for biological microscopy images.

    Enhancements:
    - Preserve input scale and sensitivity for small structures.
    - Allow tightly-clustered inliers to pass if enough inliers are present.
    - Fallback to ECC (translation/affine) registration when homography fails
      but there is some valid local overlap.
    """

    def __init__(self, detector='sift', ratio=0.85, ransac_thresh=3.0, min_inliers=8,
                 do_clahe=False, denoise_ksize=1, min_spread_frac=0.0005, try_use_gpu=False):
        self.detector_name = detector.lower()
        self.ratio = float(ratio)
        self.ransac_thresh = float(ransac_thresh)
        self.min_inliers = int(min_inliers)
        self.do_clahe = bool(do_clahe)
        self.denoise_ksize = int(denoise_ksize) if denoise_ksize is not None else 1
        self.min_spread_frac = float(min_spread_frac)

        # runtime flag for whether a CUDA detector was actually created
        self.using_cuda = False

        # determine whether GPU should be attempted using cached device count
        cuda_device_count = _get_cuda_device_count_cached()
        self.try_use_gpu = bool(try_use_gpu) and cuda_device_count > 0

        # log\-once flag for CUDA usage
        self._logged_cuda = False

        # detector/matcher creation is handled by _create_detector_and_matcher
        self._create_detector_and_matcher()

    def _create_detector_and_matcher(self):
        if hasattr(cv2, 'SIFT_create') and self.detector_name == 'sift':
            try:
                if self.try_use_gpu and hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'SIFT_create'):
                    self.detector = cv2.cuda.SIFT_create(nfeatures=10000)
                    self.using_cuda = True
                    logger.info("Using CUDA SIFT detector.")
                else:
                    try:
                        self.detector = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01, edgeThreshold=5)
                    except TypeError:
                        self.detector = cv2.SIFT_create(nfeatures=10000)
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
                self.is_binary = False
            except Exception:
                logger.info("Failed to create SIFT, falling back to ORB.")
                self.detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, edgeThreshold=15)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                self.is_binary = True
        elif hasattr(cv2, 'AKAZE_create') and self.detector_name in ('akaze', 'kaze'):
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.is_binary = True
        else:
            self.detector = cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, edgeThreshold=15)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.is_binary = True

    def _preprocess(self, img):
        """Enhanced preprocessing for microscopy images."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if self.do_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        if self.denoise_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.denoise_ksize, self.denoise_ksize), 0)

        return gray

    def _spatial_spread_ok(self, pts, img_shape, inlier_count=None):
        if pts is None or len(pts) == 0:
            return False
        pts = np.asarray(pts).reshape(-1, 2)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        bbox_area = max(1.0, (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]))
        img_area = max(1.0, img_shape[1] * img_shape[0])
        frac = bbox_area / img_area
        if frac >= self.min_spread_frac:
            return True
        if inlier_count is not None:
            if inlier_count >= max(self.min_inliers, 12):
                return True
        return False

    def _validate_transform_on_matches(self, H, pts1, pts2, max_mean_err=10.0):
        if H is None or len(pts1) == 0 or len(pts2) == 0:
            return float('inf')
        pts2_h = np.concatenate([pts2.reshape(-1, 2), np.ones((pts2.shape[0], 1))], axis=1)
        proj = (H @ pts2_h.T).T
        proj_xy = (proj[:, :2].T / proj[:, 2]).T
        diffs = np.linalg.norm(proj_xy - pts1.reshape(-1, 2), axis=1)
        mean_err = float(np.mean(diffs))
        return mean_err if mean_err <= max_mean_err else float('inf')

    def detect_and_match(self, img1, img2):

        if not self._logged_cuda:
            logger.info("Detector using CUDA: %s", bool(getattr(self, 'using_cuda', False)))
            self._logged_cuda = True

        gray1 = self._preprocess(img1)
        gray2 = self._preprocess(img2)
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return {'success': False, 'H': None, 'inliers': 0, 'matches': [], 'kp1': kp1, 'kp2': kp2, 'mask': None}

        try:
            knn = self.matcher.knnMatch(des1, des2, k=2)
        except Exception:
            try:
                bf = cv2.BFMatcher(cv2.NORM_L2 if not self.is_binary else cv2.NORM_HAMMING)
                knn = bf.knnMatch(des1, des2, k=2)
            except Exception:
                return {'success': False, 'H': None, 'inliers': 0, 'matches': [], 'kp1': kp1, 'kp2': kp2, 'mask': None}

        good = []
        for m_n in knn:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if not getattr(self.matcher, 'crossCheck', False):
            try:
                knn_rev = self.matcher.knnMatch(des2, des1, k=2)
                good_rev = set()
                for m_n in knn_rev:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < self.ratio * n.distance:
                        good_rev.add((m.trainIdx, m.queryIdx))
                good = [m for m in good if (m.queryIdx, m.trainIdx) in good_rev]
            except Exception:
                pass

        if len(good) < 4:
            return {'success': False, 'H': None, 'inliers': 0, 'matches': good, 'kp1': kp1, 'kp2': kp2, 'mask': None}

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC,
                                              ransacReprojThreshold=self.ransac_thresh)

        if H is not None:
            H_full = np.eye(3, dtype=np.float64)
            H_full[:2, :] = H
            H = H_full

        inliers = int(mask.sum()) if mask is not None else 0
        inlier_pts1 = pts1[mask.ravel() == 1] if mask is not None else np.empty((0, 1, 2))
        spread_ok = self._spatial_spread_ok(inlier_pts1, gray1.shape, inlier_count=inliers)

        success = (H is not None) and (inliers >= self.min_inliers) and spread_ok

        return {'success': success, 'H': H, 'inliers': inliers, 'matches': good, 'kp1': kp1, 'kp2': kp2, 'mask': mask}


class SimpleStitcher:
    """
    SimpleStitcher with multi-band blending and exposure compensation.
    """

    def __init__(self, detector='sift', ratio=0.75, ransac_thresh=4.0, min_inliers=30,
                 warper_type='plane', crop=True, try_use_gpu=False, skip_reorder=True,
                 blender_type='multiband', exposure_compensation=False, num_bands=5,
                 equalize=False):
        self.matcher = FeatureMatcher(detector=detector, ratio=ratio, ransac_thresh=ransac_thresh,
                                      min_inliers=min_inliers, try_use_gpu=try_use_gpu)
        self.warper_type = (warper_type or 'plane').lower()
        self.crop = bool(crop)
        self.skip_reorder = bool(skip_reorder)
        self.blender_type = blender_type.lower()
        self.exposure_compensation = bool(exposure_compensation)
        self.num_bands = int(num_bands)
        self.equalize = bool(equalize)  # boolean flag for equalization

    @staticmethod
    def _cylindrical_warp(img, f=None):
        h, w = img.shape[:2]
        if f is None:
            f = max(w, h)
        cx = w / 2.0
        cy = h / 2.0
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                x_c = (x - cx) / f
                y_c = (y - cy) / f
                denom = np.sqrt(1 + x_c * x_c)
                x_p = np.arctan(x_c)
                X = np.sin(x_p)
                Z = np.cos(x_p)
                u = f * X / Z + cx
                v = f * y_c / Z + cy
                map_x[y, x] = u
                map_y[y, x] = v
        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.remap(np.ones((h, w), dtype=np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT)
        return warped, mask

    @staticmethod
    def adjust_background_brightness(images, target_percentile=90):
        """
        Adjust brightness of images to match the darkest background brightness.
        Uses a fixed percentile to robustly identify background levels,
        avoiding bias from specimen size (unlike Otsu's method).
        """
        if not images:
            return []

        background_levels = []
        for img in images:
            if img is None or img.ndim != 3:
                background_levels.append(0)
                continue

            # Work in LAB space to isolate luminance
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, _, _ = cv2.split(lab)

            # Use a high percentile to estimate the background level (assumed bright)
            # This is much more stable than Otsu's when specimen area varies.
            bg_level = np.percentile(l, target_percentile)
            background_levels.append(bg_level)

        # Find the darkest background among all images to use as target
        # This prevents overexposure/clipping during adjustment.
        min_bg_level = min(background_levels)
        
        # Calculate a robust average of background levels to avoid bias from 
        # a single extremely dark image (e.g. if one image is mostly specimen).
        # We use a weighted target towards the median to be more stable.
        median_bg_level = np.median(background_levels)
        target_bg_level = (min_bg_level + median_bg_level) / 2.0
        
        logger.info(f"Background normalization: target level set to {target_bg_level:.2f} (min: {min_bg_level:.2f}, median: {median_bg_level:.2f})")

        adjusted_images = []
        for img, current_bg in zip(images, background_levels):
            if img is None or img.ndim != 3:
                adjusted_images.append(img)
                continue

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Scale L channel to match the global target background level
            scale = target_bg_level / current_bg if current_bg > 0 else 1.0

            # Apply scaling safely
            l_adjusted = cv2.convertScaleAbs(l, alpha=scale, beta=0)

            lab_adjusted = cv2.merge([l_adjusted, a, b])
            adjusted_images.append(cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR))

        return adjusted_images

    @staticmethod
    def auto_spread_brightness(img_bgr, low_percentile=5, high_percentile=95):
        """
        Spread brightness like ACR Auto by stretching L channel percentiles to full range.
        """
        if img_bgr is None or img_bgr.ndim != 3:
            return img_bgr
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Compute percentiles for stretching
        low_val = np.percentile(l, low_percentile)
        high_val = np.percentile(l, high_percentile)
        # Stretch L to 0-255
        l_stretched = np.clip((l - low_val) / (high_val - low_val + 1e-6) * 255, 0, 255).astype(np.uint8)
        lab_stretched = cv2.merge([l_stretched, a, b])
        return cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _estimate_vignette_mask(img_gray, sigma=0.2):
        """
        Estimate a simple Gaussian vignette mask for an image.
        Assumes the center is brightest and falls off towards edges.
        """
        h, w = img_gray.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w * sigma)
        kernel_y = cv2.getGaussianKernel(h, h * sigma)
        kernel = kernel_y @ kernel_x.T
        mask = kernel / kernel.max()
        return mask

    def _apply_vignette_correction(self, imgs, sigma=1.0):
        """
        Apply a simple inverse Gaussian mask to compensate for vignetting.
        Uses a more conservative sigma (1.0) to avoid over-correction.
        """
        corrected = []
        for img in imgs:
            if img is None:
                corrected.append(img)
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_float = l.astype(np.float32)
            
            mask = self._estimate_vignette_mask(l, sigma=sigma)
            # Inverse mask to boost edges. 
            # We use a power of the mask to make the correction more subtle at edges
            # and avoid "halo" effects or over-correction.
            correction = 1.0 / (mask + 0.1)
            correction = correction / correction.min()
            
            # Damp the correction: only apply 50% of the calculated boost to stay safe
            correction = 1.0 + (correction - 1.0) * 0.5
            
            l_corr = np.clip(l_float * correction, 0, 255).astype(np.uint8)
            lab_corr = cv2.merge([l_corr, a, b])
            corrected.append(cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR))
        return corrected

    def _equalize_images(self, imgs):
        # return [self.apply_clahe_bgr(img) if img is not None else None for img in imgs]
        imgs = self.adjust_background_brightness(imgs)
        # Apply a very mild vignette correction to reduce internal gradients
        return self._apply_vignette_correction(imgs, sigma=1.0)

    def _estimate_exposure_gains(self, imgs, Hs, min_overlap_pixels=100):
        """
        Compute per-image multiplicative gains using background-only intensity ratios
        in overlapping regions to prevent specimen density from biasing brightness.
        """
        if len(imgs) == 0:
            return []

        gains = [1.0]

        for i in range(1, len(imgs)):
            h1, w1 = imgs[i - 1].shape[:2]
            h2, w2 = imgs[i].shape[:2]

            # 1. Create coverage masks
            mask1 = np.ones((h1, w1), dtype=np.uint8) * 255
            mask2 = np.ones((h2, w2), dtype=np.uint8) * 255

            # 2. Compute relative homography
            try:
                H_rel = np.linalg.inv(Hs[i - 1]) @ Hs[i]
            except Exception:
                H_rel = np.eye(3, dtype=np.float64)

            # 3. Find overlap region
            warped_mask2 = cv2.warpPerspective(mask2, H_rel, (w1, h1), flags=cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            overlap_mask = (mask1 > 0) & (warped_mask2 > 0)

            gain = 1.0
            if overlap_mask.sum() > min_overlap_pixels:
                # Warp second image into first image coordinates for direct comparison
                warped_img2 = cv2.warpPerspective(imgs[i], H_rel, (w1, h1),
                                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                # Convert to grayscale for luminance comparison
                gray1 = cv2.cvtColor(imgs[i - 1], cv2.COLOR_BGR2GRAY) if imgs[i - 1].ndim == 3 else imgs[i - 1]
                gray2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) if warped_img2.ndim == 3 else warped_img2

                # 4. Background-Aware Filtering
                # We extract pixels from the overlap and find the "background" threshold.
                # In brightfield, the background is the brightest part (high percentile).
                pix1_all = gray1[overlap_mask]
                pix2_all = gray2[overlap_mask]

                if pix1_all.size > 0 and pix2_all.size > 0:
                    # Determine background threshold (e.g., 70th percentile)
                    bg_thresh1 = np.percentile(pix1_all, 70)
                    bg_thresh2 = np.percentile(pix2_all, 70)

                    # Create masks for pixels that are "background" in BOTH images
                    bg_mask1 = (gray1 >= bg_thresh1) & overlap_mask
                    bg_mask2 = (gray2 >= bg_thresh2) & overlap_mask
                    combined_bg_mask = bg_mask1 & bg_mask2

                    if combined_bg_mask.sum() > min_overlap_pixels:
                        # Calculate gain based ONLY on background pixels
                        median1 = float(np.median(gray1[combined_bg_mask]))
                        median2 = float(np.median(gray2[combined_bg_mask]))
                        gain = median1 / (median2 + 1e-6)
                    else:
                        # Fallback: if no clear background overlap, use whole overlap medians
                        median1 = float(np.median(pix1_all))
                        median2 = float(np.median(pix2_all))
                        gain = median1 / (median2 + 1e-6)

            gains.append(gains[-1] * gain)

        # 5. Normalize gains to prevent global drift
        mean_gain = float(np.mean(gains)) if len(gains) > 0 else 1.0
        if mean_gain == 0: mean_gain = 1.0
        gains = [g / mean_gain for g in gains]

        return gains

    def _apply_exposure_compensation(self, imgs, gains, gain_min=0.7, gain_max=1.3):
        """
        Apply global exposure gains in LAB L-channel only, clamped to a safe range.
        This reduces visible brightness gradients in the stitched result.
        """
        compensated = []

        # Clamp gains to avoid over/under compensation that creates strong gradients
        gains = [max(gain_min, min(gain_max, float(g))) for g in gains]

        for img, gain in zip(imgs, gains):
            if img is None:
                compensated.append(img)
                continue

            # Work in LAB to change only luminance
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply gain to L channel only
            l_float = l.astype(np.float32)
            l_adj = np.clip(l_float * gain, 0, 255).astype(np.uint8)

            lab_adj = cv2.merge([l_adj, a, b])
            compensated_img = cv2.cvtColor(lab_adj, cv2.COLOR_LAB2BGR)

            compensated.append(compensated_img)

        return compensated

    def _multiband_blend(self, imgs, Hs, canvas_size):
        """Multi-band blending for seamless transitions."""
        canvas_h, canvas_w = canvas_size

        # Build Laplacian pyramids for each image
        pyramids = []
        weight_pyramids = []

        # Calculate canvas offset
        corners = []
        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            corners.append(dst.reshape(-1, 2))
        all_pts = np.vstack(corners)
        x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
        offset = np.array([-x_min, -y_min])

        # Combined coverage mask to ensure unmapped areas stay black
        combined_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            trans = H.copy()
            trans[:2, 2] += offset

            # Warp image and create mask
            warped = cv2.warpPerspective(img, trans, (canvas_w, canvas_h))
            mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, trans, (canvas_w, canvas_h))

            # Update combined coverage mask
            combined_mask = ((combined_mask > 0) | (mask > 0)).astype(np.uint8)

            # Create distance transform for smooth weighting
            mask_float = mask.astype(np.float32) / 255.0
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            # Use a slightly less aggressive falloff (1.2 instead of 1.5) 
            # to balance between edge suppression and transition smoothness.
            weight = np.power(np.clip(dist / (dist.max() + 1e-6), 0, 1), 1.2)

            # Build Laplacian pyramid for image
            img_pyramid = [warped.astype(np.float32)]
            for level in range(self.num_bands - 1):
                warped = cv2.pyrDown(warped.astype(np.float32))
                img_pyramid.append(warped)

            # Convert to Laplacian
            laplacian_pyramid = [img_pyramid[-1]]
            for level in range(self.num_bands - 1, 0, -1):
                size = (img_pyramid[level - 1].shape[1], img_pyramid[level - 1].shape[0])
                expanded = cv2.pyrUp(img_pyramid[level], dstsize=size)
                laplacian = cv2.subtract(img_pyramid[level - 1], expanded)
                laplacian_pyramid.append(laplacian)
            laplacian_pyramid.reverse()

            # Build Gaussian pyramid for weight
            weight_pyramid = [weight]
            temp_weight = weight
            for level in range(self.num_bands - 1):
                temp_weight = cv2.pyrDown(temp_weight)
                weight_pyramid.append(temp_weight)

            pyramids.append(laplacian_pyramid)
            weight_pyramids.append(weight_pyramid)

        # Blend pyramids
        blended_pyramid = []
        for level in range(self.num_bands):
            blended_level = np.zeros_like(pyramids[0][level])
            weight_sum = np.zeros(pyramids[0][level].shape[:2], dtype=np.float32)

            for pyr, wpyr in zip(pyramids, weight_pyramids):
                w = wpyr[level]
                if w.shape != pyr[level].shape[:2]:
                    w = cv2.resize(w, (pyr[level].shape[1], pyr[level].shape[0]))

                w_3ch = w[:, :, np.newaxis] if pyr[level].ndim == 3 else w
                blended_level += pyr[level] * w_3ch
                weight_sum += w

            weight_sum = np.maximum(weight_sum, 1e-6)
            weight_sum_3ch = weight_sum[:, :, np.newaxis] if blended_level.ndim == 3 else weight_sum
            blended_level /= weight_sum_3ch
            blended_pyramid.append(blended_level)

        # Reconstruct from pyramid
        result = blended_pyramid[-1]
        for level in range(self.num_bands - 2, -1, -1):
            size = (blended_pyramid[level].shape[1], blended_pyramid[level].shape[0])
            result = cv2.pyrUp(result, dstsize=size)
            result = cv2.add(result, blended_pyramid[level])

        result = np.clip(result, 0, 255).astype(np.uint8)

        # 4. Force unmapped canvas pixels to black (0,0,0)
        if combined_mask is not None:
            unseen = (combined_mask == 0)
            result[unseen] = 0

        # 5. Final Gradient Smoothing (Optional but helpful for large panoramas)
        # We can apply a very mild blurring to the transitions if needed, 
        # but multi-band blending already handles most of it.
        
        return result

    @staticmethod
    def _warp_images_to_canvas(imgs, Hs):
        """Simple alpha blending fallback."""
        corners = []
        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            trans = np.eye(3) if H is None else H
            dst = cv2.perspectiveTransform(pts, trans)
            corners.append(dst.reshape(-1, 2))
        all_pts = np.vstack(corners)
        x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
        offset = np.array([-x_min, -y_min])
        canvas_w = max(1, x_max - x_min)
        canvas_h = max(1, y_max - y_min)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        acc_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            trans = np.eye(3) if H is None else H.copy()
            trans[:2, 2] += offset
            warped = cv2.warpPerspective(img, trans, (canvas_w, canvas_h))
            mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8), trans, (canvas_w, canvas_h))
            overlap = (acc_mask > 0) & (mask > 0)
            only_new = (mask > 0) & (~overlap)
            canvas[only_new == 1] = warped[only_new == 1]
            if overlap.any():
                alpha = 0.5
                canvas[overlap] = (canvas[overlap].astype(np.float32) * (1.0 - alpha) + warped[overlap].astype(
                    np.float32) * alpha).astype(np.uint8)
            acc_mask = ((acc_mask > 0) | (mask > 0)).astype(np.uint8)
        return canvas

    @staticmethod
    def _crop_black_borders(img, threshold=20, border=10):
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # Inward-trim by `border` pixels on each side
        x2 = int(max(0, x + border))
        y2 = int(max(0, y + border))
        x_end = int(min(img.shape[1], x + w - border))
        y_end = int(min(img.shape[0], y + h - border))

        # If trimming would produce an invalid box, fall back to the original bounding rect
        if x_end <= x2 or y_end <= y2:
            return img[y:y + h, x:x + w]

        return img[y2:y_end, x2:x_end]

    def stitch(self, filelist, img_loader=lambda p: cv2.imread(p, cv2.IMREAD_COLOR), assume_same_scale=True):
        """Enhanced stitching with multi-band blending and exposure compensation."""
        if not filelist:
            return None
        imgs = [img_loader(p) for p in filelist]
        if any(im is None for im in imgs):
            raise ValueError("One or more images failed to load.")

        # log matcher GPU usage at runtime
        logger.info("Matcher using CUDA: %s", bool(getattr(self.matcher, 'using_cuda', False)))

        # apply optional pre-stitch equalization before matching
        if self.equalize:
            imgs = self._equalize_images(imgs)
            logger.info("Pre-equalized brightness.")

        # Sequential chaining with affine transforms
        Hs = [np.eye(3, dtype=np.float64)]
        H_prev = np.eye(3, dtype=np.float64)

        for i in range(1, len(imgs)):
            prev = imgs[i - 1]
            cur = imgs[i]

            res = self.matcher.detect_and_match(prev, cur)

            if res.get('success'):
                H_pair = res.get('H', np.eye(3, dtype=np.float64))
                logger.info(f"Matched images {i - 1} <-> {i} with {res['inliers']} inliers")
            else:
                logger.warning(f"Failed to match images {i - 1} <-> {i}, using estimated offset")
                tx = prev.shape[1] * 0.7
                H_pair = np.array([[1, 0, tx], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

            H_cur_to_pano = H_prev @ H_pair
            Hs.append(H_cur_to_pano)
            H_prev = H_cur_to_pano

        # Apply exposure compensation if enabled
        if self.exposure_compensation:
            gains = self._estimate_exposure_gains(imgs, Hs)
            imgs = self._apply_exposure_compensation(imgs, gains)
            logger.info(f"Applied exposure compensation with gains: {gains}")

        # Calculate canvas size
        corners = []
        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            corners.append(dst.reshape(-1, 2))
        all_pts = np.vstack(corners)
        x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
        canvas_w = max(1, x_max - x_min)
        canvas_h = max(1, y_max - y_min)

        # Apply blending
        if self.blender_type == 'multiband' and len(imgs) > 1:
            logger.info(f"Using multi-band blending with {self.num_bands} bands")
            pano = self._multiband_blend(imgs, Hs, (canvas_h, canvas_w))
        else:
            logger.info("Using simple alpha blending")
            pano = self._warp_images_to_canvas(imgs, Hs)

        if self.crop:
            pano = self._crop_black_borders(pano)

        # pano = self.auto_spread_brightness(pano)
        return pano
