import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def create_custom_colormap():
    colors = [(0, 0, 1),      # Blue for negative maxima
             (0, 0, 0),     # White for intermediate negative
             (1, 0, 0)]       # Red for positive maxima
    
    positions = [0, 0.5, 1]
    
    return LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)))

def ray_gaussian_2d(x, u, sigma, phi):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x + u * np.tan(phi)) ** 2) / (2 * sigma ** 2))

def compute_second_derivative(x_vals, u_vals, sigma, phi):
    x, u = np.meshgrid(x_vals, u_vals)
    kernel = ray_gaussian_2d(x, u, sigma, phi)
    second_derivative = np.gradient(np.gradient(kernel, axis=0), axis=0)
    return second_derivative

# Parameters
x_range = 256  # Reduced for sharper visualization
u_range = 256  # Made square to match reference
x_vals = np.linspace(-x_range//2, x_range//2, x_range)
u_vals = np.linspace(-u_range//2, u_range//2, u_range)

# Updated sigma values
sigma_values = np.array([4, 6, 8, 10, 12, 14, 16])

# Updated phi range from -pi/3 to pi/3
phi_values = np.linspace(-np.pi/3, np.pi/3, 8)

# Create figure grid
n_sigma = len(sigma_values)
n_phi = len(phi_values)

# Create custom colormap
custom_cmap = create_custom_colormap()

# Create the figure
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Rows: Sigma values, Columns: Phi values', fontsize=16)

# Plot second derivatives only
for i, sigma in enumerate(sigma_values):
    sigma = sigma * 3
    for j, phi in enumerate(phi_values):
        ax = fig.add_subplot(n_sigma, n_phi, i*n_phi + j + 1)
        second_derivative = compute_second_derivative(x_vals, u_vals, sigma, phi)
        
        # Normalize the second derivative
        max_val = np.max(np.abs(second_derivative))
        second_derivative = second_derivative / max_val if max_val != 0 else second_derivative
        
        # Set symmetric limits and use extent to properly align the image
        extent = [-x_range//2, x_range//2, -u_range//2, u_range//2]
        im = ax.imshow(second_derivative, cmap=custom_cmap, aspect='equal',
                      vmin=-1, vmax=1, extent=extent, interpolation='nearest')
        
        if i == 0:
            ax.set_title(f'φ = {phi:.2f}')
        if j == 0:
            ax.set_ylabel(f'σ = {sigma}')
        ax.set_xticks([])
        ax.set_yticks([])

# Add colorbar
plt.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Normalized Second Derivative')

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save the figure
plt.savefig(os.path.join('output', 'ray_gaussian_second_derivatives.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Ray Gaussian second derivatives grid saved at: output/ray_gaussian_second_derivatives.png')
