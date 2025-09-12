# numpy_linear_algebra_basics.py
# A script to demonstrate fundamental linear algebra operations using NumPy.

import numpy as np
from scipy.sparse import diags

print("--- 1. Basic Matrix and Vector Creation ---")
# A is a 3x4 matrix (3 rows, 4 columns)
A = np.array([[1, 2, -10, 4], [3, 4, 5, -6], [3, 3, -2, 5]])
# B is a 1x4 vector (1 row, 4 columns)
B = np.array([3, 3, 4, 2])
print("Matrix A:\n", A)
print("Vector B:\n", B)

print("\n--- 2. Getting the 'Length' (Max Dimension) ---")
# The shape of a matrix is (rows, cols). max(matrix.shape) gives the larger dimension.
length_A = max(A.shape)
length_B = max(B.shape)
print("Length of A:", length_A)
print("Length of B:", length_B)

print("\n--- 3. Stacking B onto A ---")
# np.vstack "vertically stacks" matrices. B is added as a new row to A.
C = np.vstack((A, B))
print("Matrix C (A with B stacked as a new row):\n", C)

print("\n--- 4. Slicing a Matrix ---")
# Create matrix D from C.
# C[1:4, 2:4] means:
# rows: from index 1 up to (but not including) 4 -> (rows 2, 3, 4)
# columns: from index 2 up to (but not including) 4 -> (columns 3, 4)
D = C[1:4, 2:4]
print("Matrix D (rows 2-4, cols 3-4 of C):\n", D)

print("\n--- 5. Transposing a Matrix ---")
# The .T attribute transposes the matrix (rows become columns and vice versa).
E = D.T
print("Matrix E (Transpose of D):\n", E)

print("\n--- 6. Checking the Size (Shape) of a Matrix ---")
# .shape returns a tuple (number_of_rows, number_of_columns)
rows_E, cols_E = E.shape
print("Matrix E has", rows_E, "rows and", cols_E, "columns.")

print("\n--- 7. Creating Equally Spaced Vectors ---")
# np.arange creates values with a defined step size.
# np.linspace creates a specific number of values within a range.
vec_arange = np.arange(0, 2 * np.pi, np.pi / 10)
vec_linspace = np.linspace(0, 2 * np.pi, 21) # Note: 21 points to include both 0 and 2*pi
print("Vector using arange (step size pi/10):\n", vec_arange)
print("Vector using linspace (21 points):\n", vec_linspace)

print("\n--- 8 & 9. Finding Max/Min Values in Matrix A ---")
# axis=0 operates on each column.
# axis=1 operates on each row.
# No axis argument operates on the entire matrix.
max_in_cols_A = np.max(A, axis=0)
min_in_cols_A = np.min(A, axis=0)
max_in_rows_A = np.max(A, axis=1)
min_in_rows_A = np.min(A, axis=1)
max_total_A = np.max(A)
min_total_A = np.min(A)
print("Max of each column:", max_in_cols_A)
print("Min of each column:", min_in_cols_A)
print("Max of each row:", max_in_rows_A)
print("Min of each row:", min_in_rows_A)
print("Overall Max of A:", max_total_A)
print("Overall Min of A:", min_total_A)

print("\n--- 10. Statistical Operations on Matrix A ---")
mean_of_cols_A = np.mean(A, axis=0)
sum_of_rows_A = np.sum(A, axis=1)
mean_total_A = np.mean(A)
sum_total_A = np.sum(A)
print("Mean of each column:", mean_of_cols_A)
print("Sum of each row:", sum_of_rows_A)
print("Overall Mean of A:", f"{mean_total_A:.2f}")
print("Overall Sum of A:", sum_total_A)

print("\n--- 11. Creating Random Matrices ---")
# Create 5x3 matrices with random integers from -4 to 4 (inclusive).
F = np.random.randint(low=-4, high=5, size=(5, 3))
G = np.random.randint(low=-4, high=5, size=(5, 3))
print("Random Matrix F:\n", F)
print("Random Matrix G:\n", G)

print("\n--- 12. Matrix Arithmetic ---")
# These are element-wise operations.
scalar_mult_F = 0.4 * F
sum_FG = F + G
diff_FG = F - G
element_prod_FG = F * G # Note: This is NOT matrix multiplication
print("Scalar Multiplication (0.4 * F):\n", scalar_mult_F)
print("Addition (F + G):\n", sum_FG)
print("Element-wise Product (F * G):\n", element_prod_FG)

print("\n--- 13 & 14. Matrix Multiplication ---")
# For matrix multiplication (F @ A), the number of columns in F
# must equal the number of rows in A.
size_F = F.shape # (5, 3)
size_A = A.shape # (3, 4)
print("Shape of F:", size_F)
print("Shape of A:", size_A)
if size_F[1] == size_A[0]:
    print("Dimensions are compatible for multiplication (3 == 3).")
    H = F @ A # The @ operator performs matrix multiplication
    print("Resulting Matrix H (F @ A):\n", H)
    print("Shape of H:", H.shape) # Resulting shape is (5, 4)
else:
    print("Cannot multiply F and A due to incompatible dimensions.")

print("\n--- 15, 16, 17. Creating Special Matrices ---")
# np.eye() for identity matrix
identity_3x3 = np.eye(3)
# np.zeros() and np.ones()
zeros_5x3 = np.zeros((5, 3))
ones_4x2 = np.ones((4, 2))
# np.diag() for a diagonal matrix from a list
diag_S = np.diag([1, 2, 7])
print("3x3 Identity Matrix:\n", identity_3x3)
print("5x3 Matrix of Zeros:\n", zeros_5x3)
print("Diagonal Matrix S:\n", diag_S)

print("\n--- 18. Extracting Diagonal Elements ---")
# Create a random 6x6 matrix
R = np.random.rand(6, 6)
# np.diag() extracts the diagonal if given a 2D matrix
diag_R = np.diag(R)
print("Diagonal elements of a random 6x6 matrix:\n", diag_R)

print("\n--- 19. Creating a Sparse Matrix ---")
# Creates a 10x10 matrix that is mostly zeros, but has values on its main
# diagonal and the diagonals immediately above and below it.
# This is memory-efficient for very large matrices.
sparse_diag_matrix = diags([-1, 2, -1], [-1, 0, 1], shape=(10, 10)).toarray()
print("10x10 Sparse Diagonal Matrix (Tridiagonal):\n", sparse_diag_matrix)