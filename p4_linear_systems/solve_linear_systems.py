# solve_linear_systems.py
#
# A script to demonstrate and compare various numerical methods for solving
# systems of linear equations of the form Ax = b.

import numpy as np
import scipy.linalg
import time
from sympy import Matrix # Used for a robust RREF calculation

def main():
    """Main function to run all demonstrations."""
    
    print("--- Part 1-3: Solving a Square System (Ax = b) ---")
    
    # 1. Define a 5x5 matrix A and a column vector b
    A = np.array([
        [17, 24,  1,  8, 15],
        [23,  5,  7, 14, 16],
        [ 4,  6, 13, 20, 22],
        [10, 12, 19, 21,  3],
        [11, 18, 25,  2,  9]
    ], dtype=float)
    b = np.array([10, 26, 42, 59, 38], dtype=float)

    print("Matrix A:\n", A)
    print("\nVector b:\n", b)

    # 2. Solve Ax = b using the standard, high-performance solver
    x_solve = np.linalg.solve(A, b)
    print("\nSolution (x) using np.linalg.solve:\n", x_solve)

    # 3. Calculate the residual (error): r = Ax - b. Should be close to zero.
    residual = A @ x_solve - b
    print("\nResidual (A*x - b):\n", residual)
    print("-" * 50)


    print("\n--- Part 4: Solving with LU Decomposition ---")
    # Factor A into P*L*U (Permutation, Lower-triangular, Upper-triangular)
    P, L, U = scipy.linalg.lu(A)
    # The solution should be identical to np.linalg.solve
    x_lu = np.linalg.solve(A, b)
    print("Permutation Matrix P:\n", P)
    print("\nLower-Triangular Matrix L:\n", L)
    print("\nUpper-Triangular Matrix U:\n", U)
    print("\nSolution (x) from LU is the same as solve:\n", x_lu)
    print("-" * 50)

    
    print("\n--- Part 5 & 6: Solving with Least Squares and Inverse ---")
    # For a square, invertible matrix, lstsq gives the same exact solution
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    print("Solution (y) using Least Squares (lstsq):\n", x_lstsq)

    # Solve using the matrix inverse: x = A^(-1) * b
    # Note: This is less efficient and numerically stable than linalg.solve
    A_inv = np.linalg.inv(A)
    x_inv = A_inv @ b
    inv_error = x_solve - x_inv # Compare to the standard solution
    print("\nSolution (x) using Matrix Inverse:\n", x_inv)
    print("\nError between solve() and inverse() methods:\n", inv_error)
    print("-" * 50)
    

    print("\n--- Part 7: Solving with Reduced Row Echelon Form (RREF) ---")
    # Augment matrix A with vector b to form [A|b]
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    
    # Use sympy for a robust RREF calculation
    rref_matrix, pivots = Matrix(augmented_matrix).rref()
    rref_matrix = np.array(rref_matrix).astype(float)
    
    x_rref = rref_matrix[:, -1]
    rref_error = x_solve - x_rref
    print("RREF of augmented matrix [A|b]:\n", rref_matrix)
    print("\nSolution (x) from RREF:\n", x_rref)
    print("\nError between solve() and RREF methods:\n", rref_error)
    print("-" * 50)
    

    print("\n--- Part 8: Performance Comparison ---")
    # Create a large, well-conditioned matrix and vector
    N = 500
    A_large = np.random.rand(N, N) + N * np.eye(N)
    b_large = np.random.rand(N, 1)

    # Method 1: linalg.solve (fastest and most stable)
    start_time = time.time()
    x1 = np.linalg.solve(A_large, b_large)
    time_solve = time.time() - start_time

    # Method 2: Matrix inverse (slower)
    start_time = time.time()
    A_inv_large = np.linalg.inv(A_large)
    x2 = A_inv_large @ b_large
    time_inv = time.time() - start_time

    print(f"Time for np.linalg.solve ({N}x{N}): {time_solve:.6f} seconds")
    print(f"Time for matrix inverse method ({N}x{N}): {time_inv:.6f} seconds")
    print("Note: RREF is too slow for a direct comparison on large matrices.")
    print("-" * 50)

    
    print("\n--- Part 9: Overdetermined System (More Equations than Unknowns) ---")
    # 4 equations, 3 unknowns. No exact solution exists.
    A_over = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [9, 11, 12]])
    b_over = np.array([1, 2, 3, 4]).reshape(-1, 1)
    
    # lstsq finds the 'best fit' solution that minimizes the error ||Ax - b||
    x_best_fit = np.linalg.lstsq(A_over, b_over, rcond=None)[0]
    residual_over = A_over @ x_best_fit - b_over
    
    print("Overdetermined Matrix A:\n", A_over)
    print("\nVector b:\n", b_over.flatten())
    print("\nBest fit solution (x) using lstsq:\n", x_best_fit.flatten())
    print("\nResidual vector (error):\n", residual_over.flatten())
    print("-" * 50)


    print("\n--- Part 10: Underdetermined System (Fewer Equations than Unknowns) ---")
    # 3 equations, 4 unknowns. Infinitely many solutions exist.
    A_under = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b_under = np.array([1, 3, 5]).reshape(-1, 1)
    
    # lstsq finds the particular solution with the minimum norm ||x||
    x_particular = np.linalg.lstsq(A_under, b_under, rcond=None)[0]
    
    # The general solution is x_p + Z*c where Z is the null space of A
    null_space_A = scipy.linalg.null_space(A_under)
    
    print("Underdetermined Matrix A:\n", A_under)
    print("\nVector b:\n", b_under.flatten())
    print("\nParticular solution (x) from lstsq (minimum norm):\n", x_particular.flatten())
    print("\nBasis for the Null Space of A:\n", null_space_A)
    print("\nGeneral solution is x + (any linear combination of null space vectors)")
    print("-" * 50)

if __name__ == '__main__':
    main()
