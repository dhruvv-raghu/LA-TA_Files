# 👨‍💻 Project 11: Facial Recognition with Eigenfaces

This project contains a Python script (`eigenfaces.py`) that implements a basic facial recognition system using the **Eigenfaces** method. This technique, which is a direct application of **Principal Component Analysis (PCA)**, is a foundational algorithm in computer vision for tasks like facial recognition and reconstruction.

---
## 🔬 The Core Concept: Principal Component Analysis (PCA)

At a high level, PCA is a technique for **dimensionality reduction**. A high-resolution grayscale image can be thought of as a vector in a very high-dimensional space (e.g., a 100x100 image is a vector with 10,000 dimensions). PCA helps us find a much smaller set of dimensions that still captures most of the important information or variance in the data.

The Eigenfaces algorithm applies PCA to a database of faces:

1.  **Represent Faces as Vectors:** Each image in the training database is "unrolled" from a 2D matrix into a single column vector. These vectors are collected into a large matrix.
2.  **Calculate the "Mean Face":** The average of all face vectors is computed. This "mean face" is then subtracted from every face vector to center the data. 
3.  **Find the Principal Components:** PCA is used to find the principal components of the centered face vectors. These components are the eigenvectors of the data's covariance matrix. When reshaped back into an image, these eigenvectors look like ghostly faces, which are called **"Eigenfaces"**. They represent the fundamental features (e.g., variations in lighting, expression, and facial structure) across the entire training set.
4.  **Create a "Face Space":** The top eigenvectors (those corresponding to the largest eigenvalues) form a new, lower-dimensional basis, often called "face space."
5.  **Recognition and Reconstruction:** A new, unknown face can be recognized by projecting its vector onto this face space. This projection gives us a set of weights that describe how to build the new face using a combination of the Eigenfaces. By comparing these weights to the weights of known faces in the database, we can find the closest match. We can also use these weights to reconstruct an approximation of the new face using only our Eigenfaces.

---
## 🚀 How to Run the Script

### 1. Prerequisites
You need Python and the `numpy`, `matplotlib`, and `Pillow` libraries.

### 2. Install Libraries
If you don't have them, open your terminal and run:
```bash
pip install numpy matplotlib Pillow
```

### 3. Data Files
1.  Make sure you have a folder named `materials`
2.  Inside `materials`, there should be a `database` folder containing the training and testing face images (e.g., `person1.pgm`, `person31.pgm`, etc.).

The file structure should look like this:
```
.
├── p11_eigenfaces/
    ├── eigenfaces.py
    ├── README.md
└── materials/
    └── database/
        ├── person1.pgm
        ├── person2.pgm
        ...
        └── person33.pgm
```

### 4. Execute the Script
Run the script from your terminal:
```bash
python eigenfaces.py
```
The script will load the training images, calculate and display the mean face and the Eigenfaces, and then attempt to reconstruct three new faces from the database.