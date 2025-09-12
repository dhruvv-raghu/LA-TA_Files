# ⚡ Project 14: Singular Value Decomposition (SVD) in Action

This project contains a Python script (`svd_applications.py`) that explores the **Singular Value Decomposition (SVD)**, one of the most important and useful matrix factorizations in linear algebra.

The script provides a visual demonstration of what SVD does to a linear transformation and showcases two of its most famous applications: **image compression** and **image denoising**.

---
## 🔬 The Core Concept: Singular Value Decomposition

SVD states that any rectangular matrix **A** can be broken down into the product of three other matrices:
$$
A = U \Sigma V^T
$$
[Image of the SVD formula matrices]

Where:
-   **U**: An **orthogonal matrix** whose columns ($u_i$) are the "left singular vectors." It represents a **rotation** (and possibly a reflection) in the output space (codomain).
-   **$\Sigma$ (Sigma)**: A **diagonal matrix** whose diagonal entries ($\sigma_i$) are the "singular values" of A. It represents a **scaling** operation along the new axes. The singular values are always non-negative and are sorted in descending order.
-   **V**: An **orthogonal matrix** whose columns ($v_i$) are the "right singular vectors." Its transpose, $V^T$, represents a **rotation** (and possibly a reflection) in the input space (domain).

Geometrically, SVD tells us that any linear transformation **A** can be understood as a sequence of three fundamental operations:
1.  A **rotation** ($V^T$).
2.  A **scaling** along the new axes ($\Sigma$).
3.  Another **rotation** ($U$).

### Applications
-   **Image Compression:** A digital image is just a matrix of pixel values. The largest singular values in $\Sigma$ correspond to the most significant features of the image. By keeping only the top *k* singular values (and the corresponding vectors in U and V), we can create a lower-rank approximation of the image that requires much less data to store.
-   **Image Denoising:** Random noise in an image tends to be distributed across many of the smaller singular values. By discarding these smaller values (setting them to zero) and reconstructing the image, we can effectively remove a significant amount of the noise while preserving the main structure of the image.

---
## 🚀 How to Run the Script

### 1. Prerequisites
You need Python and the `numpy`, `matplotlib`, and `opencv-python` libraries.

### 2. Install Libraries
If you don't have them, open your terminal and run:
```bash
pip install numpy matplotlib opencv-python
```

### 3. Data Files
1.  Make sure you have a folder named `materials`
2.  Place any grayscale images you want to test inside this `materials` folder (e.g., `Albert_Einstein_Head.jpg`, `checkers.pgm`).

The file structure should look like this:
```
.
├── p14_svd_applications/
│   └── svd_applications.py
└── materials/
    ├── Albert_Einstein_Head.jpg
    └── checkers.pgm
```

### 4. Execute the Script
Run the script from your terminal:
```bash
python svd_applications.py
```
You will need to **edit the script** to choose which images you want to use for the compression and denoising demonstrations. The script will generate and display several plots illustrating the SVD process and its applications.