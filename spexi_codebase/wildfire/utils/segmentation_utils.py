
import numpy as np


def calculate_pixels_per_color(mask, palette):
    """
    Calculate the number of pixels for each color in the mask.

    Args:
        mask (numpy.ndarray): The color mask image (H x W x 3).
        palette (numpy.ndarray): Array of RGB colors representing the palette.

    Returns:
        dict: A dictionary mapping each color (class) to its pixel count.
    """
    mask_reshaped = mask.reshape(-1, 3)  # Flatten the mask for easier processing
    pixel_counts = {}

    for idx, color in enumerate(palette):
        count = np.sum(np.all(mask_reshaped == color, axis=1))
        pixel_counts[tuple(color)] = count  # Use the color tuple as the key

    return pixel_counts


def assign_risk_level(flammability_percentage, risk_levels):
    for level, details in risk_levels.items():
        if details["range"][0] <= flammability_percentage <= details["range"][1]:
            return details["color"], level
    return [0, 0, 0], "Unknown"


def subtract_overlapping_pixels(color_mask, palette, target_class_name, classes):
    """
    Subtracts pixels of all other classes from the target class in the color mask.

    Parameters:
    - color_mask: The color mask as a NumPy array.
    - palette: A NumPy array representing the color palette for the classes.
    - target_class_name: The name of the class to subtract from (e.g., 'Alive Tree').
    - classes: A tuple of class names corresponding to the palette.

    Returns:
    - updated_color_mask: The updated color mask with overlaps removed from the target class.
    """
    import cv2
    import numpy as np

    # Create a dictionary mapping class names to their colors
    class_colors = {class_name: color for class_name, color in zip(classes, palette)}

    # Create a binary mask for the target class
    target_color = class_colors[target_class_name]
    target_mask = (color_mask == target_color).all(axis=-1).astype(np.uint8)

    # Create a combined mask for all other classes
    combined_other_classes_mask = np.zeros_like(target_mask, dtype=np.uint8)

    for class_name, color in class_colors.items():
        if class_name != target_class_name:  # Exclude the target class
            class_mask = (color_mask == color).all(axis=-1).astype(np.uint8)
            combined_other_classes_mask = cv2.bitwise_or(combined_other_classes_mask, class_mask)

    # Subtract overlapping pixels of other classes from the target class
    exclusive_target_mask = cv2.subtract(target_mask, combined_other_classes_mask)

    # Update the color mask to reflect the subtraction
    updated_color_mask = color_mask.copy()
    updated_color_mask[exclusive_target_mask == 0] = class_colors["bg"]  # Assign background color to removed pixels

    # Keep all other class pixels intact
    for class_name, color in class_colors.items():
        if class_name != target_class_name:
            class_mask = (color_mask == color).all(axis=-1)
            updated_color_mask[class_mask] = color  # Reassign the original color

    return updated_color_mask
