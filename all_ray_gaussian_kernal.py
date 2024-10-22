import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = 'output'

# Parameters
sigma = 60
phi = np.pi / 4  # Angle phi = 45 degrees

# Create grid for the kernel
x_range = 768
u_range = 90

# Define the Ray Gaussian kernel function
def ray_gaussian_2d(x, u, sigma, phi):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x + u * np.tan(phi)) ** 2) / (2 * sigma ** 2))

x_vals = np.linspace(-x_range//2, x_range//2, x_range)
u_vals = np.linspace(-u_range//2, u_range//2, u_range)
x, u = np.meshgrid(x_vals, u_vals)

# Compute the Ray Gaussian kernel
kernel = ray_gaussian_2d(x, u, sigma, phi)

# Compute the second derivative of the kernel along the x direction
# Approximate using second-order finite difference
second_derivative = np.gradient(np.gradient(kernel, axis=0), axis=0)

# Normalize the second derivative for visualization
second_derivative_normalized = (second_derivative - np.min(second_derivative)) / (np.max(second_derivative) - np.min(second_derivative))

# Create a figure with 4 subplots
fig = plt.figure(figsize=(12, 12))

# 2D Ray Gaussian kernel
ax1 = fig.add_subplot(2, 2, 1)
im1 = ax1.imshow(kernel, cmap='gray', extent=[-x_range//2, x_range//2, -u_range//2, u_range//2])
ax1.set_title('2D Ray Gaussian Kernel')
ax1.set_xlabel('x')
ax1.set_ylabel('u')
plt.colorbar(im1, ax=ax1)  # Correct way to add colorbar

# 2D Second derivative of Ray Gaussian kernel
ax2 = fig.add_subplot(2, 2, 2)
im2 = ax2.imshow(second_derivative_normalized, cmap='gray', extent=[-x_range//2, x_range//2, -u_range//2, u_range//2])
ax2.set_title('2D Second Derivative Ray Gaussian Kernel')
ax2.set_xlabel('x')
ax2.set_ylabel('u')
plt.colorbar(im2, ax=ax2)  # Correct way to add colorbar

# 3D Ray Gaussian kernel
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(x, u, kernel, cmap='viridis')
ax3.set_title('3D Ray Gaussian Kernel')
ax3.set_xlabel('x')
ax3.set_ylabel('u')
ax3.set_zlabel('Amplitude')

# 3D Second derivative of Ray Gaussian kernel
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(x, u, second_derivative_normalized, cmap='viridis')
ax4.set_title('3D Second Derivative Ray Gaussian Kernel')
ax4.set_xlabel('x')
ax4.set_ylabel('u')
ax4.set_zlabel('Amplitude')

plt.tight_layout()
# save image
saved_path = os.path.join(OUTPUT_DIR, 'all_ray_gaussian_kernels.png')
plt.savefig(saved_path)
