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
| `detector` | string | `sift` | Feature detector algorithm (`sift`, `orb`, etc.). |

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