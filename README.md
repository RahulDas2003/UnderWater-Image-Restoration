# Underwater Image Enhancement Tool

[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![C++](https://img.shields.io/badge/C++-11%2B-red.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**A real-time, interactive tool for enhancing underwater imagery using physics-based color restoration and dehazing techniques.**

This project provides a complete pipeline to correct color casts, remove haze, and improve contrast in underwater photos. It includes both a user-friendly **Python (PyQt5) GUI** version and a high-performance **C++ (OpenCV)** version.

---

## 📸 Results




| **Original Input** | **Enhanced Output** |
| :---: | :---: |
| ![Original Image](figs/8_img_.png) | ![Enhanced Result](figs/enhanced_result.jpg) |




---
## 🎥 Video Demo

Here is a real-time comparison.

<table>
  <tr>
    <th width="50%">Original Video Input</th>
    <th width="50%">Enhanced Video Output</th>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/3f9de515-fb8e-4348-a4de-d375a558b343" controls="controls" muted="muted" style="max-width: 100%;"></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/c0dbb4f3-cc7f-4fd2-97c9-c7bde30167f9
" controls="controls" muted="muted" style="max-width: 100%;"></video>
    </td>
  </tr>
</table>

> **Note**: If the video does not play, please try using a different browser or check the file path.

## 🖼️ Dataset

This project uses the **UIEB (Underwater Image Enhancement Benchmark)** dataset for testing and validation. The raw dataset preserves the original underwater lighting conditions, making it ideal for evaluating restoration algorithms.

* **Dataset Name**: UIEB Dataset (Raw)
* **Download Link**: [Kaggle - UIEB Dataset Raw](https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw)
* **Description**: Contains raw underwater images covering various water types (greenish, bluish, turbid).

**How to use:**
1.  Download the dataset from the Kaggle link above.
2.  Extract the images into a local folder.
3.  Use the "Load Image" button in the Python tool to test these images.

---

## ✨ Features 

The enhancement pipeline consists of 6 modular stages designed to tackle specific underwater degradation issues:

1.  **White Balance Correction (LAB)**: Corrects color casts by adjusting A (Green-Red) and B (Blue-Yellow) channels in LAB space.
2.  **Red Channel Restoration**: Compensates for the severe absorption of red light using histogram equalization.
3.  **CLAHE**: Contrast-Limited Adaptive Histogram Equalization for improving local contrast without amplifying noise.
4.  **Dark Channel Dehazing**: Removes underwater haze/fog using the Dark Channel Prior (DCP) algorithm.
5.  **Adaptive Sharpening**: Unsharp masking to enhance edge details.
6.  **Gamma Correction**: Brightens shadow regions to reveal hidden details.
7.  **🆕 Auto-Tune Per Frame**: Automatic coordinate-search optimization of parameters on a per-frame basis (see below).

---

## 🔧 Auto-Tune: Automatic Per-Frame Parameter Tuning

When the **"Auto-tune parameters per frame"** checkbox is enabled, the tool automatically searches for the best combination of white-balance shifts, red boost, CLAHE clip limit, and dehaze omega for every frame.

### How It Works

| Aspect | Detail |
| --- | --- |
| **Algorithm** | Coordinate descent (one parameter at a time) starting from the current slider positions |
| **Proxy resolution** | Each frame is downscaled to ~160 px wide before scoring, keeping the search fast |
| **Iterations** | Up to 2 full passes over all 5 parameters; stops early if no improvement is found |
| **Scoring** | A no-reference image quality metric (higher is better) |

**Scoring function components:**

| Component | Weight | Direction | Purpose |
| --- | --- | --- | --- |
| Luminance contrast (σ of L channel) | +1.0 | ↑ | Reward visible contrast |
| Detail (mean \|Laplacian\| of L) | +0.5 | ↑ | Reward sharpness / fine texture |
| Color cast (chrominance distance from neutral) | −1.5 | ↓ | Penalize residual green / blue cast |
| Excess saturation (mean S above 130) | −0.8 | ↓ | Penalize garish over-saturation |
| Clipping (fraction of pixels at 0 or 255) | −80.0 | ↓ | Penalize blown highlights / crushed shadows |

### Usage

1. Set the **sliders** to reasonable starting values — they serve as the *seed* for the search.
2. Check **"Auto-tune parameters per frame"**.
3. The labels next to each slider will show the auto-selected values with an `[auto]` tag.
   - The sliders themselves are **not** moved; they continue to act as starting seeds.
4. The tuned parameters are applied to:
   - **Image preview** (Enhanced panel)
   - **Saved images** (full-resolution re-tuning)
   - **Video preview** (per-frame tuning during playback)
   - **Exported video** (per-frame tuning during export)

### Limitations

- **Speed**: Auto-tuning adds ~20–60 ms per frame on a modern CPU. Video playback may drop to a lower frame-rate for high-resolution input.
- **Search depth**: The coordinate search explores a coarse grid (step sizes of 4–10) with at most 2 iterations. It may miss the global optimum, particularly for images that need large parameter shifts away from the seed values.
- **Proxy fidelity**: Scoring is performed on a 160 px-wide thumbnail. Fine spatial details that only appear at full resolution are not captured by the scorer.
- **No temporal smoothing**: Each video frame is tuned independently, so parameters may fluctuate between consecutive frames. For very smooth results, manual parameter selection may be preferable.
- **Score heuristic**: The quality metric is a hand-tuned weighted sum; it works well for typical underwater scenes but may not suit every image type.

---

## 🛠️ Quick Start

### Option 1: Python Version (GUI)
*Best for interactive testing and parameter tuning.*

**Requirements:**
```bash
pip install opencv-python numpy PyQt5

```

**Run:**

```bash
python underwater_enhance.py

```

**User Guide:**

1. Click **"Load Image"** to select a file.
2. Adjust sliders (Omega, CLAHE, Shifts) to see real-time changes.
3. Click **"Save Result"** to export the enhanced image.

### Option 2: C++ Version (High Performance)

*Best for batch processing or embedded integration.*

**Requirements:**

* C++ Compiler (GCC, Clang, MSVC)
* OpenCV 4.x (Development headers)

**Compile (Linux/Mac with g++):**

```bash
g++ underwater_enhance.cpp -o EnhanceApp `pkg-config --cflags --libs opencv4`

```

**Compile (Windows with CMake):**
A standard `CMakeLists.txt` is recommended for linking OpenCV on Windows.

**Run:**

```bash
./EnhanceApp path/to/image.jpg

```

*(Press 'S' to save result, 'Q' to quit)*

---

## 📂 Project Structure

```text
Underwater-Image-Enhancement/
├── figs/                    # Result images for README
│   ├── 8_img_.png
│   └── enhanced_result.jpg
├── underwater_enhance.py    # Python source code with PyQt5 GUI + AutoTuner
├── underwater_enhance.cpp   # C++ source code with OpenCV HighGUI
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

```

## 📊 Pipeline Comparison

| Stage | Visual Effect | Description |
| --- | --- | --- |
| **Original** | Hazy, Blue/Green Cast | Low visibility and color distortion. |
| **WB + Red** | Color Corrected | Red component restored, color cast removed. |
| **Dehaze** | Clearer | Haze/Fog removed, transmission map applied. |
| **Final** | Sharp & Natural | Local contrast enhanced and edges sharpened. |



## 🔗 Citation

If you use this code or the original method, please acknowledge the source:

```bibtex
@misc{opencv_underwater_2025,
  title = {Guide to Underwater Image Enhancement Using OpenCV},
  author = {Samruddhi},
  year = {2025},
  howpublished = {\url{[https://opencv.org/blog/underwater-image-enhancement-using-opencv/](https://opencv.org/blog/underwater-image-enhancement-using-opencv/)}},
  note = {Accessed: 2025-12-17},
  organization = {OpenCV.org}
}
```

## 📄 License

This project is open source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
