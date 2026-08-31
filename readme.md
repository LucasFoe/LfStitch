# LfStitch: Python Tool for Image Stitching & Panoramas

[![GitHub Repository](https://img.shields.io/badge/GitHub-LucasFoe%2FLfStitch-blue?logo=github)](https://github.com/LucasFoe/LfStitch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/LucasFoe/LfStitch/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**LfStitch** is a Python tool designed for automated image stitching and panorama generation using OpenCV. It provides intelligent feature matching, homography estimation, multiband blending, exposure compensation, and border-filling post-processing to remove unmapped black borders.

Repository URL: [https://github.com/LucasFoe/LfStitch](https://github.com/LucasFoe/LfStitch)

---

## Features

- **Automated Feature Detection & Matching**: Powered by SIFT/ORB with robust RANSAC homography estimation.
- **Multiband Blending & Exposure Compensation**: Seamless transitions and uniform lighting between overlapping frames.
- **Intelligent Border Post-Processing**: Automatically estimates mean border color and fills unmapped/black border pixels (`outputfixed.jpg`).
- **Flexible Configuration**: Highly configurable via `stitch.ini`.
- **Cross-Platform**: Run with Python on Windows, macOS, and Linux, or as a standalone Windows executable (`stitch.exe`).

---

## Configuration (`stitch.ini`)

LfStitch reads its settings from `stitch.ini` in the working directory. Sensible defaults are used if options are omitted.

```ini
[OPTIONS]
img_dir = ./img
out_dir = ./result
final_megapix = 5
try_use_gpu = True
confidence_threshold = 0.5
output = output
fixborder = True
flatten_background = True
flatten_kernel_size = 61
detector = sift
```

### Configurable Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `img_dir` | string | `./img` | Path to the directory containing input images. |
| `out_dir` | string | `./result` | Path to the directory where stitched results will be saved. |
| `output` | string | `output` | Base filename for the output images (`output.jpg` and `outputfixed.jpg`). |
| `confidence_threshold` | float | `0.5` | Threshold for selecting image matches (determines minimum inliers). |
| `try_use_gpu` | boolean | `True` | Whether to attempt CUDA/GPU acceleration when available in OpenCV. |
| `final_megapix` | float | `5` | Resolution limit (in megapixels) for the final stitched image. |
| `fixborder` | boolean | `True` | Enables post-processing to fix and blend unmapped black borders. |
| `flatten_background` | boolean | `True` | Flat-field / background correction to eliminate illumination gradients and vignetting. |
| `flatten_kernel_size` | integer | `61` | Window size (pixels) for background estimation filter. |
| `detector` | string | `sift` | Feature detector algorithm (`sift`, `orb`, etc.). |

### Background Flattening & Flat-Field Correction

Microscopy, macro photography, and scanner image captures frequently exhibit uneven illumination, optical vignetting (darkened corners/edges), and background gradient variations between adjacent tiles.

- **`flatten_background` (`True` / `False`)**:
  - **What it does**: Enables pseudo flat-field background correction. It estimates the low-frequency illumination profile/envelope per color channel using morphological dilation (a rolling-ball equivalent) followed by Gaussian smoothing, then normalizes each image tile by dividing by this estimated profile.
  - **Why it matters**: It prevents visible seam lines and tile "checkerboard" artifacts caused by light falloff and edge shading across overlapping images.

- **`flatten_kernel_size` (integer, default: `61`)**:
  - **What it does**: Defines the structuring element diameter and Gaussian filter kernel size (in pixels) used to estimate the background envelope. Automatically coerced to an odd integer.
  - **How it works**: A rolling window of this size must be large enough to span across dark foreground specimen structures so they are filtered out during dilation, leaving behind only the smooth background illumination map.

#### Recommendations for `flatten_kernel_size`

| Use Case / Image Characteristic | Recommended Value | Notes |
| :--- | :--- | :--- |
| **Standard Microscopy & Brightfield** (Default) | `51` – `81` (Default: `61`) | Optimal for fine specimen structures (cells, spores, thin tissue, thalli) with bright, mostly uniform background. |
| **Thick / Large Specimen Features** | `101` – `201+` | If specimen features are wide or dense, increase kernel size so the dilation filter bridges across dark objects rather than sampling them as background. |
| **Fine Details / Small Tile Resolutions** (< 1-2 MP) | `31` – `51` | Smaller images require a smaller window size to preserve subtle global gradients without over-blurring. |
| **High-Resolution Tiles** (> 10-20 MP) | `101` – `251` | Scale up the kernel size proportionally to the image pixel dimensions so the filter area matches the physical structure scale. |
| **Complex Non-Microscopy Panoramas** (Landscape/Indoor) | `flatten_background = False` | Disable if the scene contains broad natural brightness variations across real physical objects rather than optical vignetting. |

---

## Directory Structure & Preconditions

Before executing the program, prepare the directory structure according to `stitch.ini`:

1. **Input Directory (`img_dir`)**:
   - Default: `./img`
   - Place all overlapping source images (e.g. `.jpg`, `.jpeg`, `.png`) to be stitched together.
2. **Output Directory (`out_dir`)**:
   - Default: `./result`
   - Generated files:
     - `output.jpg` – Raw stitched panorama.
     - `outputfixed.jpg` – Post-processed panorama with border correction applied.
3. **Log File**:
   - Execution details are recorded in `stitch.log` (automatically reset if size exceeds 10 MB).

---

## Installation & Setup

### Prerequisites
- Python 3.8+ (64-bit recommended)
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/LucasFoe/LfStitch.git
cd LfStitch
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Capturing Input Images (Best Practices)
For optimal stitching quality, capture overlapping images (around 30–50% overlap) covering the scene and keep the following settings fixed:
- **Consistent focus plane** (manual focus recommended).
- **Fixed exposure** (turn off automatic exposure / use AE lock).
- **Fixed white balance / color balance** (turn off automatic white balance).

### Running the Stitcher

#### Using Python
```bash
python src/stitch.py
```

#### Using Prebuilt Windows Executable
1. Extract `stitch.zip` or locate `dist/stitch.exe`.
2. Ensure `stitch.ini` and the input folder `img/` are present in the directory.
3. Run `stitch.exe`.

#### Cleaning Temporary Files (Windows)
To clean previous input and result images before a fresh run:
```cmd
scripts\del.cmd
```

#### Building Standalone Binary (PyInstaller)
To package into a single executable binary:
- **Windows**: Run `stitch.cmd` or:
  ```cmd
  pyinstaller stitch.spec
  ```
- **Linux / macOS**:
  ```bash
  pyinstaller --onefile --name stitch src/stitch.py
  ```

---

## Example Result

**Specimen:** *Metzgeria furcata* (L.) Dumort.

![Stitched Output Fixed](outputfixed.jpg)

---

## License

This project is open-source and licensed under the [MIT License](LICENSE).