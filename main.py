import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import linregress

def boxcount(Z, k):
    """
    Counts the number of non-empty boxes of size k x k in the image Z.
    """
    # Pad the image to make it divisible by k
    new_shape = (np.ceil(np.array(Z.shape) / k) * k).astype(int)
    padded_Z = np.zeros(new_shape, dtype=Z.dtype)
    padded_Z[:Z.shape[0], :Z.shape[1]] = Z

    # Reshape and sum
    S = padded_Z.reshape((new_shape[0]//k, k, new_shape[1]//k, k))
    S = S.sum(axis=(1,3))

    # Count non-empty boxes
    return np.count_nonzero(S)

def fractal_dimension(Z):
    """
    Calculates the fractal dimension of an image Z using the box-counting method.
    """
    # Only for 2D images
    assert(len(Z.shape) == 2)

    # Minimal dimension of image
    p = min(Z.shape)

    # List of box sizes (powers of 2)
    n = int(np.floor(np.log(p)/np.log(2)))
    sizes = 2**np.arange(n, 1, -1)

    counts = []
    for size in sizes:
        c = boxcount(Z, size)
        counts.append(c)

    counts = np.array(counts)
    sizes = np.array(sizes)

    # Filter out counts that are zero
    nonzero = counts > 0
    counts = counts[nonzero]
    sizes = sizes[nonzero]

    # Perform linear fit on log-log data using scipy.stats.linregress
    log_sizes = np.log(sizes)  # Using log(sizes) instead of log(1/sizes)
    log_counts = np.log(counts)
    slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_counts)

    return slope

def plot_pixel_intensity(Z):
    """
    Plots a 2D graph of the pixel intensity values (0 to 255) in the image.
    """
    plt.imshow(Z, cmap='gray', interpolation='nearest')
    plt.title('Pixel Intensity Graph')
    plt.colorbar(label='Pixel Intensity (0-255)')
    plt.show()

def process_image(image_path):
    """
    Loads the image, processes it as a 2D intensity map, plots the pixel values,
    and calculates the fractal dimension.
    """
    # Load and convert the image to grayscale
    image = Image.open(image_path).convert('L')

    # Convert to a NumPy array
    intensity_values = np.array(image)

    # Plot the pixel intensity graph
    plot_pixel_intensity(intensity_values)

    # Calculate the fractal dimension of the intensity graph
    fd = fractal_dimension(intensity_values)
    print(f"Fractal Dimension of the Intensity Graph: {fd}")

# Provide your image path here
image_path = 'path_to_your_image.png'  # Replace with your image path

# Process the image and calculate its fractal dimension
process_image(image_path)
