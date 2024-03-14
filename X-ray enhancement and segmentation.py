import cv2
import numpy as np
import matplotlib.pyplot as plt

from google.colab.patches import cv2_imshow
from google.colab import drive

def histeq():
    # Mount Google Drive
    drive.mount('/content/drive')

    # Access images from your Google Drive
    image_path1 = '/content/drive/MyDrive/s.jpg'
    input_image = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

    if input_image is None:
        print("Error: Could not load the image.")
        return None

    histeq = cv2.calcHist([input_image], [0], None, [256], [0, 256])
    cdf = histeq.cumsum()
    cdf_normalized = cdf / cdf[-1]
    equalization_map = np.round(cdf_normalized * 255).astype('uint8')
    equalized_image = equalization_map[input_image.flatten()]
    equalized_image = equalized_image.reshape(input_image.shape)

    plt.subplot(131)
    plt.imshow(input_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Histeq image')

    return equalized_image

def clahe(preprocessed_image):
    # Resize the image to a specific width and height (500x600 in this case)
    desired_width = 500
    desired_height = 600
    image = cv2.resize(preprocessed_image, (desired_width, desired_height))

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0)
    enhanced_image = clahe.apply(image)

    # Display the original image and enhanced image
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title('Histeq image')

    plt.subplot(132)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('CLAHE Image')

    plt.show()

    return enhanced_image

def bcet(preprocessed_image):
    # Load the input image
    input_image = preprocessed_image
    def balanced_contrast_enhancement(input_image, k=1.0):
        # Calculate histogram
        hist, bins = np.histogram(input_image.flatten(), 256, [0, 256])

        # Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Calculate the mean and standard deviation
        mean = np.mean(input_image)
        std_dev = np.std(input_image)

        # Calculate the threshold for balanced enhancement
        threshold = mean + (k * std_dev)

        # Initialize an output image
        output_image = np.zeros_like(input_image)

        # Apply contrast enhancement to pixels below the threshold
        low_region = input_image < threshold
        high_region = input_image >= threshold

        # Apply histogram equalization to the low region
        output_image[low_region] = np.interp(
            input_image[low_region], bins[:-1], cdf_normalized * 255
        )

        # Linear stretching to the high region
        min_high = np.min(input_image[high_region])
        max_high = np.max(input_image[high_region])
        output_image[high_region] = (
            (input_image[high_region] - min_high) / (max_high - min_high) * 255
        )

        return output_image.astype(np.uint8)

    # Set the enhancement parameter 'k' (you can adjust this)
    k_value = 1.0

    # Apply balanced contrast enhancement
    enhanced_image = balanced_contrast_enhancement(input_image, k_value)

    plt.subplot(131)
    plt.imshow(input_image, cmap='gray')
    plt.title('Histeq image')

    plt.subplot(132)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('BCET')

    plt.show()

    return enhanced_image

def gamma(preprocessed_image):
    # Define the gamma value (adjust as needed)
    gamma = 2.2

    # Perform gamma correction
    corrected_image = np.power(preprocessed_image / 255.0, gamma) * 255.0
    corrected_image = np.uint8(corrected_image)

    plt.subplot(131)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title('Histeq image')

    plt.subplot(132)
    plt.imshow(corrected_image, cmap='gray')
    plt.title('Gamma Image')

    plt.show()

    return corrected_image

def threshold(enhanced):
    # Load the enhanced image
    if enhanced is None:
        raise Exception("Image not found!")

    # Initialize a mask to keep track of segmented regions
    mask = np.zeros_like(enhanced)

    # Define seed points (you can choose these interactively or programmatically)
    seed_points = [(100, 100), (200, 200)]

    # Define a threshold for region growing (adjust as needed)
    threshold_value = 60

    # Region Growing function
    def region_growing(image, seed, threshold):
        h, w = image.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        stack = [seed]

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = 1

            # Check the intensity difference between the seed and the current pixel
            if abs(int(image[x, y]) - int(image[seed])) < threshold:
                mask[x, y] = 255  # Mark the pixel in the mask
                # Add neighboring pixels to the stack (8-connected)
                if x > 0:
                    stack.append((x - 1, y))
                if x < h - 1:
                    stack.append((x + 1, y))
                if y > 0:
                    stack.append((x, y - 1))
                if y < w - 1:
                    stack.append((x, y + 1))

    # Apply region growing to each seed point
    for seed in seed_points:
        region_growing(enhanced, seed, threshold_value)

    # Display the original and segmented images
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))  # Display the original image in RGB
    plt.title('Enhanced  Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')  # Display the mask as a grayscale image
    plt.title('Segmented Image')
    plt.axis('off')

    plt.show()

    return mask

def morphological(enhanced):
    # Load the image
    image = enhanced

    if image is None:
        raise Exception("Image not found!")

    gray = image

    # Apply thresholding to create a binary image (adjust threshold as needed)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to clean the binary image
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Perform connected component analysis to label and separate objects
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(opening)

    # Create an output image with different colors for each object
    output = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1, len(stats)):
        if stats[i][cv2.CC_STAT_AREA] > 100:  # Adjust the area threshold as needed
            mask = (labels == i).astype(np.uint8) * 255
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)

            # Corrected line: separate element-wise multiplication for each channel
            for channel in range(3):
                output[:, :, channel] += mask * color[channel]

    # Display the original and segmented images
    plt.figure(figsize=(12, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the original image in RGB
    plt.title('Original Image')
    plt.axis('off')

    # Segmented Image (Morphological)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title('Segmented Image (Morphological)')
    plt.axis('off')

    plt.show()
    return output



def edge(enhanced):
    # edge-based segmentation

    # Load the image
    image = enhanced

    if image is None:
        raise Exception("Image not found!")

    gray = image

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)  # Adjust the thresholds as needed

    # Display the original and segmented images
    plt.figure(figsize=(12, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the original image in RGB
    plt.title('Original Image')
    plt.axis('off')

    # Segmented Image (Edge-based)
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')  # Display the segmented image with edges
    plt.title('Segmented Image (Edge-based)')
    plt.axis('off')

    plt.show()
    return edges

def gaussian(preprocessed):
    # Define the image path
    image = preprocessed

    # Apply Gaussian filter
    kernel_size = (5, 5)  # Adjust the kernel size as needed
    sigma = 1.0  # Adjust the sigma value as needed
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    # Convert the image to float for logarithm calculation
    blurred_image_float = blurred_image.astype(float)

    # Calculate the logarithm of the image
    log_image = np.log1p(blurred_image_float)

    # Convert the log image back to uint8 for display
    log_image_uint8 = (255 * (log_image - log_image.min()) / (log_image.max() - log_image.min())).astype(np.uint8)

    # Display the original image, Gaussian filtered image, and log image using matplotlib

    # Original Image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Histeq')
    plt.axis('off')

    # Gaussian Filtered Image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Filtered')
    plt.axis('off')

    # Log Image
    plt.subplot(1, 3, 3)
    plt.imshow(log_image_uint8, cmap='gray')
    plt.title('Log Image')
    plt.axis('off')

    plt.show()
    return log_image_uint8

def sift(enhanced):
    # Load the image
    image = enhanced
    gray = image

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on the image
    output_image = cv2.drawKeypoints(image, keypoints, outImage=None)

    # Display the original image and SIFT image using matplotlib

    # Original Image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # SIFT Image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Image')
    plt.axis('off')

    plt.show()

    # Scale Invariant Feature Transform (SIFT)

    # Load the image
    sift_image = cv2.drawKeypoints(image, keypoints, outImage=None)
    cv2_imshow(sift_image)

def segment(enhanced_image):
    while True:
        print("\nChoose the required Segmentation Technique from the given list")
        print("1. Threshold Segmentation")
        print("2. Morphological Segmentation")
        print("3. Edge-based segmentation")
        print("4. Main Menu")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            Segmented_img = threshold(enhanced_image)
        elif choice == 2:
            Segmented_img = morphological(enhanced_image)
        elif choice == 3:
            Segmented_img = edge(enhanced_image)
        elif choice == 4:
            main()
        else:
            print("Invalid choice. Please select a valid option.")

def main():
    print("Basic histogram Equalization in process------>>>>>>>")
    print("-------------------------------------------->>>>>>>")

    print("------------------------------------------->>>>>>>")

    preprocessed_image = histeq()
    enhanced_image = None
    while True:
        print("\nChoose the required Enhancement Technique from the given list")
        print("1. CLAHE")
        print("2. BCET")
        print("3. Gamma")
        print("4. Gassian of Log")
        print("5. Apply Further Segmentation on Enhanced Image")
        print("6. Apply SIFT feature Extraction")
        print("7. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            enhanced_image = clahe(preprocessed_image)
        elif choice == 2:
            enhanced_image = bcet(preprocessed_image)
        elif choice == 3:
            enhanced_image = gamma(preprocessed_image)
        elif choice == 4:
            enhanced_image = gaussian(preprocessed_image)
        elif choice == 5:
            segment(enhanced_image)
        elif choice == 6:
            sift(enhanced_image)
        elif choice == 7:
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice... Please select a valid option...")

if __name__ == "__main__":
    main()
