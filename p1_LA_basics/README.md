# 🐍 NumPy for Linear Algebra: A Hands-On Introduction

Welcome! This repository contains a simple Python script (`numpy_linear_algebra_basics.py`) designed to introduce you to fundamental linear algebra operations using the **NumPy** library.

This script is a great starting point if you're new to Python for scientific computing or want to see how mathematical concepts from your course can be implemented in code.

---

## 🚀 How to Use This File

Follow these simple steps to get the script running on your computer.

### 1. Prerequisites

Make sure you have **Python** installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### 2. Install Libraries

You'll need two Python libraries: `numpy` for numerical operations and `scipy` for scientific computing (we use it for one example).

Open your terminal or command prompt and run the following command to install them:
```bash
pip install numpy scipy
```

### 3. Run the Script

1.  **Download the File:** Save the `numpy_linear_algebra_basics.py` file to a folder on your computer.
2.  **Navigate to the Folder:** Open your terminal and use the `cd` command to go to the directory where you saved the file. For example:
    ```bash
    cd Linear Algebra/p1_LA_basics
    ```
3.  **Execute the Script:** Run the script using the following command:
    ```bash
    python numpy_linear_algebra_basics.py
    ```

You will see the output of each operation printed directly to your terminal.

### 4. Experiment!

The best way to learn is by doing. Open the `.py` file in a text editor or IDE (like VS Code) and try changing some of the values in the matrices `A` and `B`. Rerun the script and see how the output changes!

---

## 🔬 What's Inside the Script?

The script is organized into sections, each demonstrating a different concept. You will find examples of:

* **Matrix Creation:** Defining matrices and vectors.
* **Matrix Manipulation:** Slicing (selecting parts of a matrix), stacking, and transposing.
* **Inspecting Properties:** Finding the dimensions, shape, and size of a matrix.
* **Vector & Matrix Generation:** Creating special arrays like `np.linspace`, `np.eye` (identity matrix), `np.zeros`, and `np.ones`.
* **Statistical Operations:** Calculating the min, max, mean, and sum of matrix elements (for entire matrices, columns, or rows).
* **Matrix Arithmetic:** Performing scalar multiplication, addition, subtraction, and element-wise multiplication.
* **Matrix Multiplication:** The "dot product" multiplication of two matrices.