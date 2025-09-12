# eigenfaces.py
#
# A script to demonstrate facial recognition and reconstruction using the
# Eigenfaces method, an application of Principal Component Analysis (PCA).

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# --- CORE FUNCTIONS ---

def load_training_images(database_path, database_size):
    """Loads all training images and converts them into a data matrix."""
    image_vectors = []
    for i in range(1, database_size + 1):
        try:
            img_path = os.path.join(database_path, f'person{i}.pgm')
            img = Image.open(img_path)
            # Get dimensions from the first image
            if i == 1:
                m, n = np.array(img).shape
            # Reshape image to a column vector and append
            image_vectors.append(np.array(img).reshape(m * n, 1))
        except FileNotFoundError:
            print(f"Warning: Could not find training image person{i}.pgm")
            continue
            
    # Combine all column vectors into a single matrix P
    P = np.hstack(image_vectors)
    return P, m, n

def calculate_eigenfaces(P, num_eigenfaces=None):
    """Calculates the mean face and eigenfaces from the data matrix."""
    # Step 1: Calculate the mean face
    mean_face = np.mean(P, axis=1, keepdims=True)
    
    # Step 2: Center the data by subtracting the mean face
    A = P.astype(np.float64) - mean_face
    
    # Step 3: Compute covariance matrix C = A * A.T.
    # For efficiency, we use the "snapshot" method and compute L = A.T * A
    L = A.T @ A
    
    # Step 4: Find eigenvectors and eigenvalues of L
    eigenvalues, eigenvectors_L = np.linalg.eig(L)
    
    # Step 5: Get eigenvectors of C by mapping back: U = A * V
    eigenvectors_C = A @ eigenvectors_L
    
    # Step 6: Normalize the eigenvectors to be unit vectors
    norms = np.linalg.norm(eigenvectors_C, axis=0)
    normalized_eigenvectors = eigenvectors_C / norms
    
    # Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = normalized_eigenvectors[:, sorted_indices]
    
    if num_eigenfaces:
        return mean_face, sorted_eigenvectors[:, :num_eigenfaces]
    
    return mean_face, sorted_eigenvectors

def reconstruct_face(image_path, mean_face, eigenfaces, dims):
    """Projects a new face onto the eigenface space and reconstructs it."""
    m, n = dims
    
    # Load and reshape the new image
    img = Image.open(image_path)
    original_img_array = np.array(img)
    U = original_img_array.reshape(m * n, 1)
    
    # Center the new face vector
    U_centered = U.astype(np.float64) - mean_face
    
    # Project the centered face onto the eigenface space to get weights
    weights = eigenfaces.T @ U_centered
    
    # Reconstruct the face by combining eigenfaces with their weights
    U_approx = eigenfaces @ weights + mean_face
    
    # Reshape back to image dimensions
    reconstructed_img_array = U_approx.reshape(m, n).astype(np.uint8)
    
    return original_img_array, reconstructed_img_array

def plot_results(original, reconstructed, title_original, title_reconstructed):
    """Helper function to display original and reconstructed images side-by-side."""
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title_original)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(title_reconstructed)
    plt.axis('off')
    plt.show()

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    DATABASE_SIZE = 30
    
    try:
        # Construct the path relative to the script's location
        database_path = '../materials/database'
        
        # 1. Load training data
        P, m, n = load_training_images(database_path, DATABASE_SIZE)
        print(f"Loaded {P.shape[1]} training images of size {m}x{n}.")
        
        # 2. Calculate the mean face and eigenfaces
        mean_face, eigenfaces = calculate_eigenfaces(P)
        print(f"Calculated {eigenfaces.shape[1]} eigenfaces.")

        # Display the mean face
        plot_results(mean_face.reshape(m,n), mean_face.reshape(m,n), "Mean Face", "")
        plt.close() # Close the single plot figure

        # Display the top 15 eigenfaces
        top_eigenfaces_img = np.hstack([ef.reshape(m, n) for ef in eigenfaces.T[:15]])
        plt.figure(figsize=(15, 5))
        plt.imshow(top_eigenfaces_img, cmap='gray')
        plt.title('Top 15 Eigenfaces')
        plt.axis('off')
        plt.show()

        # 3. Reconstruct new faces not in the training set
        print("\nReconstructing new faces...")
        test_images = ['person31.pgm', 'person32.pgm', 'person33.pgm']
        
        for img_name in test_images:
            test_image_path = os.path.join(database_path, img_name)
            original, reconstructed = reconstruct_face(test_image_path, mean_face, eigenfaces, (m, n))
            plot_results(original, reconstructed, f'Original: {img_name}', 'Reconstructed')

    except FileNotFoundError:
        print(f"Error: The database directory was not found at '{database_path}'.")
        print("Please ensure your project structure is correct.")

```