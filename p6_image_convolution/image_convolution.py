# image_convolution.py
#
# A script to demonstrate various image filtering techniques using convolution
# with different kernels.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# --- HELPER FUNCTIONS ---

def apply_convolution_color(image, kernel):
    """Applies a 2D convolution kernel to each channel of a color image."""
    # Ensure image is float for convolution calculations
    img_float = image.astype(np.float64)
    result = np.zeros_like(img_float)
    
    # Apply convolution to each channel (R, G, B) separately
    for channel in range(3):
        result[:, :, channel] = convolve2d(
            img_float[:, :, channel], kernel, mode='same', boundary='symm'
        )
        
    # Clip values to the valid 0-255 range and convert back to uint8
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_convolution_gray(image_gray, kernel):
    """Applies a 2D convolution kernel to a grayscale image."""
    # Ensure image is float for convolution calculations
    img_float = image_gray.astype(np.float64)
    
    result = convolve2d(img_float, kernel, mode='same', boundary='symm')
    
    # Clip values to the valid 0-255 range and convert back to uint8
    return np.clip(result, 0, 255).astype(np.uint8)

# --- MAIN EXECUTION ---

def main():
    """Load an image and apply various convolution filters."""
    
    # 1. Load the image from the 'materials' folder
    try:
        # OpenCV loads images in BGR format, so we convert to RGB
        img_bgr = cv2.imread('materials/einstein.jpg')
        if img_bgr is None:
            raise FileNotFoundError
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except FileNotFoundError:
        print("Error: 'einstein.jpg' not found in the 'materials' folder.")
        return
        
    print(f"Image dimensions: {img_rgb.shape[0]} x {img_rgb.shape[1]}")

    # 2. Denoising: Apply blurring filters to a noisy image
    # Add random noise to a copy of the original image
    noise = 50 * (np.random.rand(*img_rgb.shape) - 0.5)
    img_noisy = np.clip(img_rgb.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Define averaging and Gaussian blur kernels
    kernel_avg = np.ones((3, 3)) / 9
    kernel_gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    
    # Apply filters
    img_denoised_avg = apply_convolution_color(img_noisy, kernel_avg)
    img_denoised_gauss = apply_convolution_color(img_noisy, kernel_gauss)

    # Display denoising results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original')
    axes[1].imshow(img_noisy)
    axes[1].set_title('Noisy')
    axes[2].imshow(img_denoised_avg)
    axes[2].set_title('Average Blur Denoise')
    axes[3].imshow(img_denoised_gauss)
    axes[3].set_title('Gaussian Blur Denoise')
    for ax in axes: ax.axis('off')
    fig.suptitle('Image Denoising with Blurring Kernels')
    plt.show()

    # 3. Sharpening: Apply sharpening filters to the grayscale image
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = apply_convolution_gray(img_gray, kernel_sharp)

    # Display sharpening results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[1].imshow(img_sharp, cmap='gray')
    axes[1].set_title('Sharpened')
    for ax in axes: ax.axis('off')
    fig.suptitle('Image Sharpening')
    plt.show()

    # 4. Edge Detection: Apply Sobel and Laplace operators
    # Sobel kernels detect horizontal and vertical edges
    kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Laplacian kernel detects edges in all directions
    kernel_laplace = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
    # Apply filters
    edges_sobel_x = apply_convolution_gray(img_gray, kernel_sobel_x)
    edges_sobel_y = apply_convolution_gray(img_gray, kernel_sobel_y)
    edges_laplace = apply_convolution_gray(img_gray, kernel_laplace)
    
    # Display edge detection results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[1].imshow(edges_sobel_x, cmap='gray')
    axes[1].set_title('Sobel Horizontal Edges')
    axes[2].imshow(edges_sobel_y, cmap='gray')
    axes[2].set_title('Sobel Vertical Edges')
    axes[3].imshow(edges_laplace, cmap='gray')
    axes[3].set_title('Laplacian Edges')
    for ax in axes: ax.axis('off')
    fig.suptitle('Edge Detection with Convolution Kernels')
    plt.show()

if __name__ == '__main__':
    main()