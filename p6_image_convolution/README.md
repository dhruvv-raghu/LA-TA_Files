# 🖼️ Image Processing with Convolution Kernels

This project contains a Python script (`image_convolution.py`) that demonstrates fundamental image processing techniques using **convolution**. It shows how applying a small matrix, called a **kernel**, to an image can achieve powerful filtering effects like blurring (denoising), sharpening, and edge detection.

---

## 💡 The Core Concept: Convolution

Convolution is the process of sliding a kernel over an image, pixel by pixel, to calculate a new value for each pixel. The kernel is a small matrix of weights. The new pixel value is the weighted sum of the original pixel's neighbors.



By changing the weights in the kernel, we can achieve different effects:

* **Blurring/Averaging Kernels:** These kernels typically contain positive values that sum to 1. They average the values of neighboring pixels, which smooths out noise and reduces detail.
    $$
    \text{Average Kernel} = \frac{1}{9}
    \begin{bmatrix}
    1 & 1 & 1 \\
    1 & 1 & 1 \\
    1 & 1 & 1
    \end{bmatrix}
    $$

* **Sharpening Kernels:** These kernels, often called "unsharp masks," emphasize the difference between a pixel and its neighbors. They typically have a large positive value in the center, surrounded by negative values.
    $$
    \text{Sharpen Kernel} =
    \begin{bmatrix}
    0 & -1 & 0 \\
    -1 & 5 & -1 \\
    0 & -1 & 0
    \end{bmatrix}
    $$

* **Edge Detection Kernels (Sobel, Laplace):** These kernels are designed to detect regions of high contrast (i.e., edges). The **Sobel** operator uses two kernels to detect horizontal and vertical edges separately, while the **Laplacian** operator detects edges in all directions.

---

## 🚀 How to Run the Script

### 1. Prerequisites
You need Python and several common data science libraries.

### 2. Install Libraries
If you don't have them, open your terminal and run:
```bash
pip install opencv-python numpy matplotlib scipy
```

### 3. Data Files
1.  Make sure you have a folder named `materials`
2.  Use your image file (e.g., `einstein.jpg`) inside this `materials` folder.

The final file structure should look like this:
```
.
├── p6_image_convolution/
│   └── image_convolution.py
└── materials/
    └── einstein.jpg
```

### 4. Execute the Script
Run the script from your terminal:
```bash
python image_convolution.py
```
The script will load the image and display the results of the various filters in separate windows.

---

## 🔬 Filters Demonstrated

* **Noise Reduction:** Applying average and Gaussian blur kernels to a noisy image to smooth it out.
* **Image Sharpening:** Using a sharpening kernel to enhance edges and details.
* **Edge Detection:** Using Sobel and Laplace operators to extract the outlines of objects in the image.