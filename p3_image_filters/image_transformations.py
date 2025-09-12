# image_transformations.py
#
# A script to demonstrate image manipulation using linear algebra with NumPy.
# This version is optimized to use vectorized operations for speed and clarity.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_transform(image, matrix):
    """Applies a 3x3 transformation matrix to an image."""
    # Ensure the image is in float format for matrix multiplication
    float_image = image.astype(float)
    
    # Apply the transformation matrix to each pixel's RGB vector
    # The @ operator is equivalent to np.dot for this purpose
    transformed_image = float_image @ matrix.T
    
    # Clip values to the valid 0-255 range and convert back to uint8
    return np.clip(transformed_image, 0, 255).astype(np.uint8)

def main():
    # --- 1. Load and Display the Image ---
    try:
        # Assumes 'image.jpg' is in the same directory as the script
        image_path = "../materials/image.jpg"
        img_original = np.array(Image.open(image_path))
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        print("Please place an image file named 'image.jpg' in the same directory as the script.")
        return

    print(f"Image loaded with dimensions (height, width, channels): {img_original.shape}")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_original)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # --- 2. Extract and Display Color Channels ---
    red_channel = img_original[:, :, 0]
    green_channel = img_original[:, :, 1]
    blue_channel = img_original[:, :, 2]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(red_channel, cmap='gray')
    axes[0].set_title('Red Channel')
    axes[1].imshow(green_channel, cmap='gray')
    axes[1].set_title('Green Channel')
    axes[2].imshow(blue_channel, cmap='gray')
    axes[2].set_title('Blue Channel')
    for ax in axes: ax.axis('off')
    plt.suptitle("Individual Color Channels")
    plt.show()

    # --- 3. Define Transformation Matrices ---
    # Note: These matrices are applied to [R, G, B] vectors.
    
    GrayMatrix = np.array([
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3]
    ])

    SepiaMatrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    # Swaps the Red and Blue channels
    PermuteMatrix = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    
    # Removes the green channel, creating a magenta tint
    DeleteGreenMatrix = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    
    # Custom matrix for a specific color adjustment
    UserMatrix = np.array([
        [0.7, 0.15, 0.15],
        [0.15, 0.7, 0.15],
        [0.15, 0.15, 0.7]
    ])

    # --- 4. Apply Linear Transformations (Filters) ---
    img_gray = apply_transform(img_original, GrayMatrix)
    img_sepia = apply_transform(img_original, SepiaMatrix)
    img_permute = apply_transform(img_original, PermuteMatrix)
    img_no_green = apply_transform(img_original, DeleteGreenMatrix)
    img_user = apply_transform(img_original, UserMatrix)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(img_gray)
    axes[0, 1].set_title('Grayscale')
    axes[0, 2].imshow(img_sepia)
    axes[0, 2].set_title('Sepia')
    axes[1, 0].imshow(img_permute)
    axes[1, 0].set_title('Permuted (R/B Swapped)')
    axes[1, 1].imshow(img_no_green)
    axes[1, 1].set_title('No Green Channel')
    axes[1, 2].imshow(img_user)
    axes[1, 2].set_title('User Matrix')
    for ax in axes.flat: ax.axis('off')
    plt.suptitle("Image Filters via Matrix Transformation")
    plt.show()

    # --- 5. Apply Non-Linear Transformations ---
    img_invert = 255 - img_original
    
    # Gamma correction (adjusts brightness non-linearly)
    gamma_light = np.power(img_original / 255.0, 0.8) * 255.0
    gamma_dark = np.power(img_original / 255.0, 1.2) * 255.0
    img_gamma_light = gamma_light.astype(np.uint8)
    img_gamma_dark = gamma_dark.astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_invert)
    axes[0].set_title('Inverted Colors')
    axes[1].imshow(img_gamma_light)
    axes[1].set_title('Gamma Light (γ=0.8)')
    axes[2].imshow(img_gamma_dark)
    axes[2].set_title('Gamma Dark (γ=1.2)')
    for ax in axes.flat: ax.axis('off')
    plt.suptitle("Non-Linear Transformations")
    plt.show()

    # --- 6. Reversing a Transformation with an Inverse Matrix ---
    # Can we get the original image back from the sepia version?
    try:
        SepiaMatrix_inv = np.linalg.inv(SepiaMatrix)
        img_reconstructed = apply_transform(img_sepia, SepiaMatrix_inv)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_original)
        axes[0].set_title('Original')
        axes[1].imshow(img_sepia)
        axes[1].set_title('Sepia')
        axes[2].imshow(img_reconstructed)
        axes[2].set_title('Reconstructed from Sepia')
        for ax in axes.flat: ax.axis('off')
        plt.suptitle("Reversing a Filter with the Inverse Matrix")
        plt.show()
    except np.linalg.LinAlgError:
        print("The SepiaMatrix is not invertible.")

if __name__ == '__main__':
    main()