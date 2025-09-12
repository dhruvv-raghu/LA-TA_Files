# 3d_graphics.py
#
# A script to demonstrate 3D object manipulation using rotation matrices
# and 2D/3D plotting with Matplotlib.

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
import itertools

# --- CORE FUNCTIONS ---

def get_rotation_matrix(theta_x, theta_y, theta_z):
    """
    Generates a combined 3D rotation matrix for rotations around the
    x, y, and z axes in that order.
    """
    # Rotation matrix around the x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    # Rotation matrix around the y-axis
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    # Rotation matrix around the z-axis
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    # Combined rotation matrix (applied as Rx, then Ry, then Rz)
    return Rz @ Ry @ Rx

# --- DEMONSTRATION SECTIONS ---

def demo_cube():
    """Defines, rotates, and projects a 3D cube."""
    print("--- Running Cube Demonstration ---")
    
    # Define cube vertices and edges (adjacency matrix)
    vertices = np.array([
        [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
        [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
    ])
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4],
        [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
    ])

    # Define rotation angles and get the rotation matrix
    rot_mat = get_rotation_matrix(np.pi / 3, np.pi / 4, np.pi / 6)
    
    # Apply rotation to the vertices
    vert_rot = vertices @ rot_mat.T

    # Plot the 2D projection (dropping the Z-coordinate)
    plt.figure(figsize=(6, 6))
    plt.axis('equal')
    plt.title('2D Projection of Rotated Cube')
    for edge in edges:
        p1 = vert_rot[edge[0]]
        p2 = vert_rot[edge[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

def demo_buckyball():
    """Generates, rotates, and plots a 3D Buckyball."""
    print("\n--- Running Buckyball Demonstration ---")
    
    # Generate Buckyball coordinates
    phi = (1 + math.sqrt(5)) / 2
    base_coords = [
        (0, 1, 3 * phi), (0, -1, 3 * phi), (0, 1, -3 * phi), (0, -1, -3 * phi),
        (1, 2 + phi, 2 * phi), (-1, 2 + phi, 2 * phi), (1, -(2 + phi), 2 * phi), (-1, -(2 + phi), 2 * phi),
        (1, 2 + phi, -2 * phi), (-1, 2 + phi, -2 * phi), (1, -(2 + phi), -2 * phi), (-1, -(2 + phi), -2 * phi),
        (2, 1 + 2 * phi, phi), (-2, 1 + 2 * phi, phi), (2, -(1 + 2 * phi), phi), (-2, -(1 + 2 * phi), phi),
        (2, 1 + 2 * phi, -phi), (-2, 1 + 2 * phi, -phi), (2, -(1 + 2 * phi), -phi), (-2, -(1 + 2 * phi), -phi)
    ]
    
    coords = []
    for c in base_coords:
        coords.extend(list(set(itertools.permutations(c))))
    coords = np.array(list(set(coords)))
    
    # Create adjacency matrix based on a bond distance of 2.0
    num_vertices = coords.shape[0]
    edges = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if np.isclose(np.linalg.norm(coords[i] - coords[j]), 2.0):
                edges[i, j] = 1

    # Define and apply rotation
    rot_mat = get_rotation_matrix(np.pi / 3, np.pi / 4, np.pi / 6)
    rotated_coords = coords @ rot_mat.T
    
    # Plot the 3D projection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Projection of Buckyball')
    for j in range(num_vertices):
        for k in range(j + 1, num_vertices):
            if edges[j, k] == 1:
                p1, p2 = rotated_coords[j], rotated_coords[k]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()

def demo_model_from_file():
    """Loads a 3D model from .mat files, rotates, and projects it."""
    print("\n--- Running 3D Model from File Demonstration ---")
    
    try:
        v = scipy.io.loadmat('../materials/v.mat')['v']
        f = scipy.io.loadmat('../materials/f.mat')['f'] - 1 # Adjust for 0-based indexing
    except FileNotFoundError:
        print("Error: Ensure 'v.mat' and 'f.mat' are in the 'materials' folder.")
        return

    # Define and apply rotation
    rot_mat = get_rotation_matrix(np.pi / 3, np.pi / 4, np.pi)
    vert_rot = v @ rot_mat.T
    
    # Plot the 2D projection
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.title("2D Projection of the 3D Model")
    for face in f:
        p1, p2, p3 = vert_rot[face[0]], vert_rot[face[1]], vert_rot[face[2]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'b-')
        plt.plot([p3[0], p1[0]], [p3[1], p1[1]], 'b-')
    plt.xlabel('X'); plt.ylabel('Y'); plt.grid(True)
    plt.show()

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    demo_cube()
    # Note: The Buckyball generation is complex and may not be perfectly accurate.
    # demo_buckyball() 
    demo_model_from_file()
```