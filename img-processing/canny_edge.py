import PIL.Image
import numpy as np
import scipy.stats as st
from math import degrees

K = 3

def saveImage(imageArray, outputName):
    # Convert array to unsigned 8-bit and save using PIL
    im = PIL.Image.fromarray(imageArray.astype(np.uint8))
    im.save(outputName)

def getWidth(imageArray):
    return imageArray.shape[1]

def getHeight(imageArray):
    return imageArray.shape[0]

def getGrayScaledPixel(rgbPixel):
    return 0.299 * rgbPixel[0] + 0.587 * rgbPixel[1] + 0.114 * rgbPixel[2]

def convertToGrayScale(imageArray):
    height, width = imageArray.shape[0], imageArray.shape[1]
    result = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            result[y, x] = getGrayScaledPixel(imageArray[y, x])
    return result

def gkern(kernlen=21, nsig=30):
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def GaussianCoefficients(sigma):
    return gkern(2 * K + 1, sigma)

def getSx():
    return np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])

def getSy():
    return np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

def calculateFilterValue(kernel, patch):
    return np.sum(kernel * patch)

def apply_filter(kernel, imageArray, k):
    """
    Applies a convolution filter to imageArray.
    The kernel is assumed to be of size (2*k+1)x(2*k+1).
    """
    height, width = imageArray.shape
    # Create an output with reduced dimensions (no padding)
    result = np.empty((height - 2 * k, width - 2 * k))
    for y in range(k, height - k):
        for x in range(k, width - k):
            patch = imageArray[y - k:y + k + 1, x - k:x + k + 1]
            result[y - k, x - k] = calculateFilterValue(kernel, patch)
    return result

def calculateMagnitudeAndDirection(X, Y):
    height, width = X.shape
    mag = np.empty((height, width))
    direction = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            gx = X[y, x]
            gy = Y[y, x]
            mag[y, x] = np.sqrt(gx**2 + gy**2)
            # Use arctan2 to avoid division by zero and get the proper angle
            angle = np.arctan2(gy, gx)
            # Convert angle from radians to degrees in [0, 180)
            angle_deg = degrees(angle) % 180
            # Quantize the angle into 4 sectors
            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg < 180):
                quant = 0  # horizontal
            elif 22.5 <= angle_deg < 67.5:
                quant = 1  # 45° diagonal
            elif 67.5 <= angle_deg < 112.5:
                quant = 2  # vertical
            else:
                quant = 3  # 135° diagonal
            direction[y, x] = quant
    return mag, direction.astype(int)

def nonMaximalSuppress(image, gdirection):
    """
    Suppresses non-maximum pixels along the gradient direction.
    """
    height, width = image.shape
    suppressed = np.copy(image)
    # Process only non-border pixels
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            direction = gdirection[y, x]
            current = image[y, x]
            if direction == 0:
                neighbors = [image[y, x - 1], image[y, x + 1]]
            elif direction == 1:
                neighbors = [image[y - 1, x + 1], image[y + 1, x - 1]]
            elif direction == 2:
                neighbors = [image[y - 1, x], image[y + 1, x]]
            elif direction == 3:
                neighbors = [image[y - 1, x - 1], image[y + 1, x + 1]]
            if any(current <= n for n in neighbors):
                suppressed[y, x] = 0
    return suppressed

def doubleThreshold(image, lowThreshold, highThreshold):
    """
    Applies double thresholding:
      - Strong edges are set to 255.
      - Weak edges are set to 75.
      - Others become 0.
    """
    strong = 255
    weak = 75
    result = np.zeros_like(image)
    result[image > highThreshold] = strong
    result[(image >= lowThreshold) & (image <= highThreshold)] = weak
    return result

def edgeTracking(image):
    """
    Performs edge tracking by hysteresis:
    Any weak pixel connected (8-connected) to a strong pixel is upgraded.
    """
    height, width = image.shape
    strong = 255
    weak = 75
    tracked = np.copy(image)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if tracked[y, x] == weak:
                if (tracked[y + 1, x] == strong or tracked[y - 1, x] == strong or
                    tracked[y, x + 1] == strong or tracked[y, x - 1] == strong or
                    tracked[y + 1, x + 1] == strong or tracked[y - 1, x - 1] == strong or
                    tracked[y + 1, x - 1] == strong or tracked[y - 1, x + 1] == strong):
                    tracked[y, x] = strong
                else:
                    tracked[y, x] = 0
    return tracked

def cannyEdgeDetector(src, sigma, lowThreshold, highThreshold):
    # Open the image and convert to grayscale
    image = PIL.Image.open(src).convert("L")
    output = np.array(image, dtype=float)
    
    # Apply Gaussian smoothing
    smoothed = apply_filter(GaussianCoefficients(sigma), output, K)
    
    # Compute gradients using Sobel filters
    gx = apply_filter(getSx(), smoothed, 1)
    gy = apply_filter(getSy(), smoothed, 1)
    gradientMagnitude, gradientDirection = calculateMagnitudeAndDirection(gx, gy)
    
    # Non-maximal suppression
    suppressed = nonMaximalSuppress(gradientMagnitude, gradientDirection)
    
    # Double thresholding
    thresholded = doubleThreshold(suppressed, lowThreshold, highThreshold)
    
    # Edge tracking by hysteresis
    edges = edgeTracking(thresholded)
    return edges

# Example usage:
canny_edge = cannyEdgeDetector('img-processing/placeholder.webp', sigma=1, lowThreshold=20, highThreshold=40)
saveImage(canny_edge, "canny_edge.png")