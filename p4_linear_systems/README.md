# 🔢 Solving Linear Systems: A Numerical Approach

This project contains a Python script (`solve_linear_systems.py`) that explores various methods for solving the fundamental linear algebra problem **Ax = b**.

It's designed for students to compare the accuracy, stability, and performance of different numerical algorithms, from direct methods like LU decomposition to iterative and least-squares approaches.

---

## 🔬 Concepts Demonstrated

The script provides hands-on examples of the following methods and concepts:

* **Direct Solution:** Using `np.linalg.solve()`, the standard, highly-optimized method for solving a square linear system.
* **LU Decomposition:** Factoring a matrix **A** into **P*L*U** (Permutation, Lower-triangular, Upper-triangular) and using it to solve **Ax = b**. This is often what `solve()` does under the hood.
* **Matrix Inverse Method:** Solving for **x** by calculating **x = A⁻¹b**. This method is shown for educational purposes but is generally less efficient and numerically stable than `solve()` or LU decomposition.
* **Reduced Row Echelon Form (RREF):** A classic textbook method for solving systems by augmenting the matrix **[A|b]** and performing Gaussian elimination.
* **Performance Comparison:** A timed benchmark comparing the speed of `solve()`, the inverse method, and RREF for a large system.
* **Overdetermined Systems:** Finding the "best fit" solution to a system with more equations than unknowns using `np.linalg.lstsq()` (least squares). This is fundamental to regression analysis.
* **Underdetermined Systems:** Finding a particular solution and the general solution to a system with fewer equations than unknowns, which has infinite solutions.

---

## 🚀 How to Run the Script

### 1. Prerequisites
Ensure you have Python installed. You will also need the NumPy and SciPy libraries, which are standard for scientific computing in Python.

### 2. Install Libraries
If you don't have them installed, open your terminal or command prompt and run:
```bash
pip install numpy scipy
```

### 3. Execute the Script
In your terminal, navigate to the folder containing the script and run:
```bash
python solve_linear_systems.py
```
The script will execute all the steps sequentially and print the results of each method, including solution vectors, error residuals, and performance timings, directly to your console.