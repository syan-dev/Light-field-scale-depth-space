import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
from matplotlib.colors import LinearSegmentedColormap
import os

OUTPUT_DIR = 'output'
INPUT_EPI = 'data/epi.png'

def create_ray_gaussian_kernel(size_x, size_u, sigma, phi):
    """
    Create a normalized second derivative Ray-Gaussian kernel
    """
    x = np.linspace(-size_x//2, size_x//2, size_x)
    u = np.linspace(-size_u//2, size_u//2, size_u)
    U, X = np.meshgrid(u, x)
    x_shifted = X + U * np.tan(phi)
    kernel = (x_shifted**2 / sigma**4 - 1/sigma**2) * \
             np.exp(-x_shifted**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def create_custom_colormap():
    """
    Create a custom colormap: red -> white -> blue
    """
    colors = ['red', 'white', 'blue']
    nodes = [0.0, 0.5, 1.0]
    return LinearSegmentedColormap.from_list('custom', list(zip(nodes, colors)))

def process_epi_image(image_path, kernel_size_x=31, kernel_size_u=31, sigma=2.0, phi=np.pi/4):
    """
    Process EPI image with Ray-Gaussian kernel
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image")
    
    kernel = create_ray_gaussian_kernel(kernel_size_x, kernel_size_u, sigma, phi)
    result = signal.convolve2d(img, kernel, mode='same', boundary='symm')
    result_norm = 2 * (result - np.min(result)) / (np.max(result) - np.min(result)) - 1
    
    return img, kernel, result_norm

def visualize_results(img, kernel, result):
    """
    Visualize results in one line: EPI, Kernel, and Result
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original EPI
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('EPI')
    axes[0].axis('off')
    
    # Kernel with custom colormap
    kernel_plot = axes[1].imshow(kernel, cmap=create_custom_colormap())
    axes[1].set_title('Normalized Second Derivative\nRay-Gaussian Kernel')
    axes[1].axis('off')
    plt.colorbar(kernel_plot, ax=axes[1])
    
    # Normalized result with custom colormap
    result_plot = axes[2].imshow(result, cmap=create_custom_colormap(), vmin=-1, vmax=1)
    axes[2].set_title('Result')
    axes[2].axis('off')
    plt.colorbar(result_plot, ax=axes[2])
    
    plt.tight_layout()
    saved_path = os.path.join(OUTPUT_DIR, 'epi_kernel_result.png')
    print(f"Saving visualization at {saved_path}")
    plt.savefig(saved_path)

def main():
    image_path = INPUT_EPI
    kernel_size_x = 31
    kernel_size_u = 31
    sigma = 6
    phi = np.pi/4  # 45 degrees
    
    try:
        img, kernel, result = process_epi_image(
            image_path,
            kernel_size_x,
            kernel_size_u,
            sigma,
            phi
        )
        visualize_results(img, kernel, result)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()