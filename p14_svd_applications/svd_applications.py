# svd_applications.py
#
# A script to demonstrate the Singular Value Decomposition (SVD) and its
# applications in visualizing linear transformations, image compression,
# and image denoising.

import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- DEMONSTRATION FUNCTIONS ---

def visualize_svd_transformation():
    """
    Shows the geometric effect of an SVD transformation step-by-step
    by transforming the unit circle.
    """
    print("--- 1. Visualizing SVD on a Linear Transformation ---")
    
    # A. Define the transformation matrix and the unit circle
    A = np.array([[2, 1], [-1, 1]])
    t = np.linspace(0, 2 * np.pi, 100)
    X = np.array([np.cos(t), np.sin(t)]) # Points on the unit circle

    # B. Perform SVD
    U, S_vals, Vt = np.linalg.svd(A)
    S = np.diag(S_vals)
    V = Vt.T

    # C. Apply each transformation step
    VX = V.T @ X       # Step 1: Rotation by V^T
    SVX = S @ VX      # Step 2: Scaling by Sigma
    USVX = U @ SVX    # Step 3: Rotation by U

    # D. Plot the results in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    titles = ['1. Unit Circle', '2. Rotated by V^T', '3. Scaled by Sigma', '4. Rotated by U (Final)']
    data_points = [X, VX, SVX, USVX]

    for i, ax in enumerate(axes.flat):
        ax.plot(data_points[i][0, :], data_points[i][1, :], 'b-')
        ax.axis('equal')
        ax.set_title(titles[i])
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_image_compression(image_path):
    """
    Loads an image, performs SVD, and shows compressed versions
    using a truncated SVD.
    """
    print(f"\n--- 2. Demonstrating Image Compression on '{image_path}' ---")
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Please check the filename and path.")
        return

    # Perform SVD
    U, S_vals, Vt = np.linalg.svd(img_gray.astype(float), full_matrices=False)

    # Show plot of singular values to see their distribution
    plt.figure(figsize=(10, 5))
    plt.plot(S_vals)
    plt.title('Singular Values (ordered)')
    plt.ylabel('Magnitude')
    plt.xlabel('Singular Value Index')
    plt.grid(True)
    plt.show()

    # Reconstruct the image with different numbers of singular values
    ranks = [20, 50, 100]
    for k in ranks:
        # Reconstruct using top k singular values/vectors
        S_trunc = np.diag(S_vals[:k])
        img_approx = U[:, :k] @ S_trunc @ Vt[:k, :]
        
        # Calculate compression percentage
        original_size = img_gray.size
        compressed_size = U[:, :k].size + S_vals[:k].size + Vt[:k, :].size
        compression_ratio = 100 * (1 - compressed_size / original_size)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_approx, cmap='gray')
        plt.title(f'Compressed Image (k = {k}, {compression_ratio:.1f}% compression)')
        plt.axis('off')
        plt.show()
        
def demonstrate_image_denoising(image_path):
    """
    Adds noise to an image and uses truncated SVD to denoise it.
    """
    print(f"\n--- 3. Demonstrating Image Denoising on '{image_path}' ---")
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Please check the filename and path.")
        return
        
    # Add random noise
    noise = 50 * (np.random.rand(*img_gray.shape) - 0.5)
    img_noisy = np.clip(img_gray.astype(float) + noise, 0, 255)
    
    # Perform SVD on the noisy image
    U, S_vals, Vt = np.linalg.svd(img_noisy, full_matrices=False)
    
    # Denoise by reconstructing with a subset of singular values
    k = 50 # Heuristic: choose a rank that captures structure but not noise
    S_trunc = np.diag(S_vals[:k])
    img_denoised = U[:, :k] @ S_trunc @ Vt[:k, :]
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(img_noisy, cmap='gray')
    axes[1].set_title('Noisy')
    axes[2].imshow(img_denoised, cmap='gray')
    axes[2].set_title(f'Denoised (k = {k})')
    for ax in axes: ax.axis('off')
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Demonstration 1: Geometric interpretation of SVD
    visualize_svd_transformation()
    
    #
    # <<< ACTION REQUIRED >>>
    # Edit the filenames below to choose images from your 'materials' folder.
    #
    
    # Demonstration 2: Image Compression
    compression_image_name = "Albert_Einstein_Head.jpg"
    compression_image_path = f"../materials/{compression_image_name}"
    demonstrate_image_compression(compression_image_path)
    
    # Demonstration 3: Image Denoising
    denoising_image_name = "checkers.pgm"
    denoising_image_path = f"../materials/{denoising_image_name}"
    demonstrate_image_denoising(denoising_image_path)