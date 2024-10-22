import numpy as np
import h5py
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import time
import argparse
import os


INPUT_IMAGE = 'data/lf.h5'
OUTPUT_DIR = 'output'


def load_light_field(filepath):
    """Load light field from H5 file and convert to YCbCr"""
    print("[Info]>>Starts at", time.ctime())
    with h5py.File(filepath, 'r') as hf:
        lf_data = np.array(hf.get('LF'))
    
    rows = lf_data.shape[0]
    cols = lf_data.shape[1]
    views = np.zeros_like(lf_data, dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            views[row, col] = lf_data[row, col]
            # cv2 image show
            # vis = cv2.cvtColor(views[row, col], cv2.COLOR_RGB2BGR)
            # save_image_path = os.path.join(OUTPUT_DIR, f'view_{row}_{col}.png')
            # cv2.imwrite(save_image_path, vis)
            # print(f"View image saved at {save_image_path}")

    return views

def create_grid_of_image(views, padding=10, highlight_row=None, border_color=(0, 255, 0)):
    """Create a grid of images for visualization"""
    rows, cols, height, width, _ = views.shape
    grid = np.ones((rows * (height + padding) + padding, cols * (width + padding) + padding, 3), dtype=np.uint8) * 255
    
    for row in range(rows):
        for col in range(cols):
            y = row * (height + padding) + padding
            x = col * (width + padding) + padding
            grid[y:y+height, x:x+width] = views[row, col]

    # if row or col is hightlighted, overlay a gray opacity all unhighlighted images
    if highlight_row is not None:
        for row in range(rows):
            if row != highlight_row:
                y = row * (height + padding) + padding
                for col in range(cols):
                    x = col * (width + padding) + padding
                    grid[y:y+height, x:x+width] = cv2.addWeighted(grid[y:y+height, x:x+width], 0.2, np.ones_like(grid[y:y+height, x:x+width]) * 128, 0.8, 0)

    # draw a rectangle around highlighted row
    if highlight_row is not None:
        y = highlight_row * (height + padding) + padding
        cv2.rectangle(grid, (padding, y), (grid.shape[1] - padding, y + height), border_color, padding)

    
    return grid


def create_stack_of_images(images, padding_x=10, padding_y=10, border_size=1, border_color=(0, 255, 0)):
    # Assuming images is a numpy array with shape (N, H, W, C)
    N, H, W, C = images.shape
    
    # Calculate the final image size to accommodate the padding and the shift
    final_width = W + (N - 1) * padding_x + 2 * border_size
    final_height = H + (N - 1) * padding_y + 2 * border_size
    
    # Initialize a base image (larger canvas with black background)
    stacked_image = np.ones((final_height, final_width, C), dtype=images.dtype)*255
    
    # Place each image on the canvas with a shift in x and y
    for i, img in enumerate(images):
        # Add a border to the image
        img_with_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, 
                                             cv2.BORDER_CONSTANT, value=border_color)
        
        # Update dimensions to account for the border
        H_with_border, W_with_border = img_with_border.shape[:2]
        
        # Calculate the x and y offset for each image
        x_offset = i * padding_x
        y_offset = i * padding_y
        
        # Place the current image with border onto the canvas at the calculated offset
        stacked_image[y_offset:y_offset + H_with_border, x_offset:x_offset + W_with_border] = img_with_border
    
    return stacked_image


def create_epi_image(images, y):
    print(images.shape)
    # create a slice cut at y axis
    epi_image = np.zeros((images.shape[0], images.shape[2], 3), dtype=np.uint8)
    print(epi_image.shape)
    for i, img in enumerate(images):
        epi_image[i] = img[y]
    return epi_image

def main():
    # parser = argparse.ArgumentParser(description="Process a light field file and display the depth map.")
    # parser.add_argument("filepath", help="Path to the light field file (H5 format)")
    # args = parser.parse_args()
    grid_padding = 50
    highlight_row = 4

    # load_light_field(args.filepath)
    views = load_light_field(INPUT_IMAGE)
    grid = create_grid_of_image(views, grid_padding, highlight_row)

    saved_grid_path = os.path.join(OUTPUT_DIR, 'grid.png')
    vis_grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    # resize smaller grid
    vis_grid = cv2.resize(vis_grid, (vis_grid.shape[1]//9, vis_grid.shape[0]//9))
    cv2.imwrite(saved_grid_path, vis_grid)
    print(f"Grid image saved at {saved_grid_path}")
    
    # get all images on row highlight_row
    images = views[highlight_row]

    # create a stack of images
    stacked_image = create_stack_of_images(images, padding_x=10, padding_y=10)

    saved_stacked_path = os.path.join(OUTPUT_DIR, 'stacked.png')
    vis_stacked = cv2.cvtColor(stacked_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(saved_stacked_path, vis_stacked)

    # create epipolar image at y is half of height and save
    y = views.shape[2] // 2
    epi_image = create_epi_image(images, y)
    saved_epi_path = os.path.join(OUTPUT_DIR, 'epi.png')
    vis_epi = cv2.cvtColor(epi_image, cv2.COLOR_RGB2BGR)
    # resize for better visualization
    vis_epi = cv2.resize(vis_epi, (vis_epi.shape[1], vis_epi.shape[0]*10))
    cv2.imwrite(saved_epi_path, vis_epi)
    print(f"EPI image saved at {saved_epi_path}")

    # create stack of first half
    # split images into two halves
    first_half = images[:, :images.shape[2]//2]
    stacked_image_first_half = create_stack_of_images(first_half, padding_x=5, padding_y=5, border_size=0)
    saved_stacked_first_half_path = os.path.join(OUTPUT_DIR, 'stacked_first_half.png')
    vis_stacked_first_half = cv2.cvtColor(stacked_image_first_half, cv2.COLOR_RGB2BGR)
    cv2.imwrite(saved_stacked_first_half_path, vis_stacked_first_half)
    print(f"Stacked first half image saved at {saved_stacked_first_half_path}")

    # second half
    second_half = images[:, images.shape[2]//2:]
    stacked_image_second_half = create_stack_of_images(second_half, padding_x=5, padding_y=5, border_size=0)
    saved_stacked_second_half_path = os.path.join(OUTPUT_DIR, 'stacked_second_half.png')
    vis_stacked_second_half = cv2.cvtColor(stacked_image_second_half, cv2.COLOR_RGB2BGR)
    cv2.imwrite(saved_stacked_second_half_path, vis_stacked_second_half)


if __name__ == "__main__":
    main()
