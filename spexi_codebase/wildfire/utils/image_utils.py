
import cv2
import numpy as np

def is_high_contrast(image, threshold=70):
    """
    Checks whether an image is high contrast based on the variance of pixel intensities.
    
    Parameters:
    - image: Input image (in BGR format).
    - threshold: Variance threshold to determine high contrast.
    
    Returns:
    - True if high contrast, False otherwise.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_image)
    return std_dev >= threshold

def increase_highlights_auto_threshold(img, highlight_amount=2, percentile=50):
    """
    Increases highlights in the image by brightening the brightest regions.
    Automatically determines a threshold based on the brightness distribution.
    
    Parameters:
    - image_path: Path to the image file.
    - highlight_amount: Multiplier for highlight brightness (e.g., 1.2 = 20% increase).
    - percentile: Percentile to use for determining the brightness threshold.
    
    Returns:
    - Processed image with increased highlights.
    """
    # Load the image
    # image = cv2.imread(image_path)
    image = img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

    # Convert to HSV (Hue, Saturation, Value) for better control over brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Calculate the percentile threshold based on brightness values
    threshold = np.percentile(v, percentile)

    # Increase brightness only for pixels above the threshold
    highlight_mask = v > threshold
    v[highlight_mask] = np.clip(v[highlight_mask] * highlight_amount, 0, 255)

    # Merge channels back and convert to RGB
    hsv_highlighted = cv2.merge([h, s, v])
    highlighted_image = cv2.cvtColor(hsv_highlighted, cv2.COLOR_HSV2RGB)

    # Save the processed image
    highlighted_image_bgr = cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    return highlighted_image

def detect_green_areas(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 20])
    upper_green = np.array([100, 255, 180])
    return cv2.inRange(hsv_image, lower_green, upper_green)
