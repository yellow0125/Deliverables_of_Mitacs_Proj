
import os
import cv2
import mmcv
import numpy as np

def save_image(output_path, image, message):
        """Helper function to save an image and print a message."""
        mmcv.imwrite(image, output_path)
        print(f"{message}: {output_path}")

def save_mask_or_visual(img, color_mask, total_relevant_pixels, overlay_color, flammability_percentage,
                        water_percentage, poi, al_percentage, risk_level, output_mode, output_subfolder, file_name):
   """
    Save the output in 'mask', 'visual', or 'segmented' mode based on the configuration.
   """

   if output_mode == 'mask':
        if total_relevant_pixels > 0 or flammability_percentage > 0 :
            # Create a mask overlay with the specified overlay color
            mask_overlay = np.zeros_like(img, dtype=np.uint8)
            mask_overlay[:, :] = overlay_color
            save_image(os.path.join(output_subfolder, file_name), mask_overlay, "Saved overlay image")
        else:
            # Save the original image if no relevant pixels are detected
            save_image(os.path.join(output_subfolder, file_name), img, "Saved original image")
   elif output_mode == 'visual':
        if total_relevant_pixels > 0 or flammability_percentage > 0:
            # Create a mask overlay and include text annotations
            mask_overlay = np.zeros_like(img, dtype=np.uint8)
            mask_overlay[:, :] = overlay_color  # Fill the mask with overlay color

            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_color = (255, 255, 255)  # White color for text
            line_spacing = 30  # Spacing between lines

            # Define the text to display
            text_lines = [
                f"Flammability Percentage: {flammability_percentage:.2f}%",
                f"Water Percentage: {water_percentage:.2f}%",
                f"Point of Interest (POI): {poi:.2f}%",
                f"Alive Tree Percentage (al): {al_percentage:.2f}%",
                f"Risk Level: {risk_level}"
            ]

            # Place the text on the mask
            for i, text in enumerate(text_lines):
                position = (10, 30 + i * line_spacing)  # Calculate text position
                cv2.putText(mask_overlay, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Save the visual overlay
            save_image(os.path.join(output_subfolder, file_name), mask_overlay, "Saved visual overlay image")
        else:
            # Save the original image if no overlay is applied
            save_image(os.path.join(output_subfolder, file_name), img, "Saved original image")
            
   elif output_mode == 'segmented':
        # Blend the color mask with the original image
        if total_relevant_pixels > 0 or flammability_percentage > 0 or water_percentage > 0 :
            alpha_trees = 0.3  # Transparency factor for blending
            blended_image = ((1 - alpha_trees) * img + alpha_trees * color_mask).astype(np.uint8)

            # Save the blended image
            save_image(os.path.join(output_subfolder, file_name), blended_image, "Saved segmented image with overlay")
        else:
            # Save the original image if no overlay is applied
            save_image(os.path.join(output_subfolder, file_name), img, "Saved original image")