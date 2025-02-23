import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import *
from PIL import Image as PILImage
from scipy.ndimage import laplace
import os
# Custom function to convert an RGB image to grayscale
def rgb2gray(rgb):
    # Uses the standard luminance conversion formula
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

# Load the image using PIL (ensure the image is in RGB)
color_image = PILImage.open('imgs/fig 2.png').convert('RGB')
Image_arr = np.array(color_image)

# Convert the RGB image to grayscale using our custom function
gray_image = rgb2gray(Image_arr)
img = np.array(gray_image, dtype=np.float64)

# Initialize the Level Set Function (LSF)
IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
IniLSF[30:80, 30:80] = -1
IniLSF = -IniLSF

# Display the initial contour on the image
plt.figure(1)
plt.imshow(Image_arr)
plt.xticks([]), plt.yticks([])  # Hide tick values
plt.contour(IniLSF, [0], color='b', linewidth=2)
plt.draw(), plt.show(block=False)


def mat_math(input_array, op):
    # This function applies a math operation elementwise (inefficient loop version).
    # For production code, consider using vectorized operations.
    output = input_array.copy()
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            if op == "atan":
                output[i, j] = math.atan(input_array[i, j])
            elif op == "sqrt":
                output[i, j] = math.sqrt(input_array[i, j])
    return output


def CV(LSF, img, mu, nu, epison, step):
    # Compute the Dirac delta approximation and Heaviside function
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan"))
    
    # Compute gradients of the LSF
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix * Ix + Iy * Iy, "sqrt")
    Nx = Ix / (s + 1e-6)
    Ny = Iy / (s + 1e-6)
    
    # Compute curvature (divergence of normalized gradient)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy
    Length = nu * Drc * cur

    # Compute Laplacian using SciPy's laplace function
    Lap = laplace(LSF)
    Penalty = mu * (Lap - cur)

    # Compute the Chan-Vese terms using local intensity means
    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) ** 2 + 1 * (img - C2) ** 2)

    # Evolve the level set function
    LSF = LSF + step * (Length + Penalty + CVterm)
    return LSF


# Set parameters
mu = 1
nu = 0.003 * 255 * 255
num = 20
epison = 1
step = 0.1
LSF = IniLSF
output_dir = 'level_set_iterations'
os.makedirs(output_dir, exist_ok=True)

# List to store frame filenames
frames = []
# Evolve the level set function over iterations
for i in range(1, num):
    LSF = CV(LSF, img, mu, nu, epison, step)
    
    # Save each iteration as a frame
    plt.imshow(Image_arr)
    plt.xticks([]), plt.yticks([])
    plt.contour(LSF, [0], colors='r', linewidth=2)
    
    # Save the frame
    frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
    plt.savefig(frame_path)
    frames.append(frame_path)
    plt.close()

images = [PILImage.open(f) for f in frames]
images[0].save('level_set_evolution.gif',
               save_all=True,
               append_images=images[1:],
               duration=200,  # Duration for each frame in milliseconds
               loop=0)  # 0 means loop forever
