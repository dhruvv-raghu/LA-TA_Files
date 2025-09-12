# 🧊 Project 9: 3D Graphics with Rotation Matrices

This project contains a Python script (`3d_graphics.py`) that demonstrates how linear algebra is used to manipulate and visualize 3D objects. It uses **rotation matrices** to change the orientation of objects in 3D space and then **projects** them onto a 2D plane to be displayed on a screen.

The script visualizes three different objects: a simple cube, a complex Buckyball (C60 molecule), and a 3D model loaded from data files.

---
## 🔬 The Core Concepts

### 1. Representing 3D Objects
A 3D object can be represented by a set of points in space called **vertices**. The connections between these vertices are called **edges** or **faces**. We can store the coordinates of all vertices in an $N \times 3$ matrix, where $N$ is the number of vertices.

### 2. 3D Rotation Matrices
To rotate a point or an entire object in 3D space, we multiply its vertex matrix by a **rotation matrix**. A rotation by an angle $\theta$ around a specific axis (X, Y, or Z) can be represented by a unique $3 \times 3$ matrix.

* **Rotation around X-axis:**
    $$
    R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}
    $$
* **Rotation around Y-axis:**
    $$
    R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}
    $$
* **Rotation around Z-axis:**
    $$
    R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}
    $$

To perform a sequence of rotations, we multiply these matrices together. For example, a combined rotation is $R = R_z R_y R_x$. Applying this to our vertex matrix `V` gives the new, rotated vertices: $V_{\text{rotated}} = V \cdot R^T$.

### 3. Orthographic Projection
To view a 3D object on a 2D screen, we must project it. The simplest method is an **orthographic projection**, where we simply drop one of the coordinates (e.g., the Z-coordinate) to plot the remaining X and Y coordinates.

---
## 🚀 How to Run the Script

### 1. Prerequisites
You need Python and the `numpy`, `scipy`, and `matplotlib` libraries.

### 2. Install Libraries
If you don't have them, open your terminal and run:
```bash
pip install numpy scipy matplotlib
```

### 3. Data Files
1.  Make sure you have a folder named `materials` in the same directory as the script.
2.  Place the `v.mat` and `f.mat` files inside this `materials` folder.

The file structure should look like this:
```
.
├── p9_3d_graphics/
│   ├── 3d_graphics.py
└── materials/
    ├── v.mat
    └── f.mat
```

### 4. Execute the Script
Run the script from your terminal:
```bash
python 3d_graphics.py
```
The script will generate and display several plots showing the rotated 3D objects.