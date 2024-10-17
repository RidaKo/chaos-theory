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
        print(f"Box size: {size}, Count: {c}")

    counts = np.array(counts)
    sizes = np.array(sizes)

    # Filter out counts that are zero or less
    nonzero = counts > 0
    counts = counts[nonzero]
    sizes = sizes[nonzero]

    # Perform linear fit on log-log data
    
    #coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    # return -coeffs[0]

    log_sizes = np.log(1/sizes)
    log_counts = np.log(counts)
    slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_counts)

    return -slope

def plot_box_counting(Z):
    p = min(Z.shape)
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

    plt.figure(figsize=(8, 6))
    plt.plot(np.log(1/sizes), np.log(counts), 'o-', mfc='none')
    plt.title('Box-Counting Method')
    plt.xlabel('log(1/ε)')
    plt.ylabel('log N(ε)')
    plt.grid(True)
    plt.show()

# Load and convert the image to grayscale
image_path = 'C:\\Users\\kroda\\OneDrive\\Chaos\\image3.png'  # Replace with your image path
image = Image.open(image_path).convert('L')

# Convert to binary image (black and white)
threshold = 128
binary_image = np.array(image) < threshold
binary_image = binary_image.astype(np.uint8)

# Check unique values
print("Unique values in binary image:", np.unique(binary_image))

# Display the binary image
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')
plt.show()

# Calculate the fractal dimension
fd = fractal_dimension(binary_image)
print(f"Fractal Dimension: {fd}")

# Plot the log-log graph for the box-counting method
plot_box_counting(binary_image)
