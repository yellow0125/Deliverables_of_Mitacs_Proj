
import os
import mmcv
import cv2
from config.config import (
    MODEL_CONFIG, MODEL_CHECKPOINT, INPUT_FOLDER, OUTPUT_FOLDER, OUTPUT_FOLDER_POI,
    TILE_LEVEL, KRPANO_TOOL_PATH,
    WATER_MODEL_PATH, SPHERE_IMAGE_PATH, HOTSPOTS_XML_PATH,  # Added paths
    CLASSES, DEVICE, PALETTE, RISK_LEVELS, OUTPUT_MODE, CONTRAST_THRESHOLD, HIGHLIGHT_AMOUNT, PERCENTILE,OUT
)
from utils.file_utils import create_output_dirs, get_relative_path, save_color_mask,update_index_js
from utils.image_utils import is_high_contrast, increase_highlights_auto_threshold, detect_green_areas
from utils.segmentation_utils import calculate_pixels_per_color, assign_risk_level, subtract_overlapping_pixels
from utils.save_mask_or_visual import save_mask_or_visual
from utils.tile_to_sphere import cube_to_sphere
from utils.hotspot_utils import generate_hotspots_and_visualize
from model_utils.process_image import ProcessImage
import shutil

print("Starting processing...")
input_folder = os.path.join(INPUT_FOLDER, str(TILE_LEVEL))
# Create output directories
output_folder = os.path.join(OUTPUT_FOLDER, str(TILE_LEVEL))
output_folder_poi = os.path.join(OUTPUT_FOLDER_POI, str(TILE_LEVEL))
create_output_dirs(output_folder, output_folder_poi)

process_image = ProcessImage()
# Process input images
for root, dirs, files in os.walk(input_folder):
    for file_name in files:
        if not file_name.lower().endswith('.jpg'):
            continue

        img_path = os.path.join(root, file_name)
        relative_path = get_relative_path(root, input_folder)
        print(f"Processing image: {img_path}")
    
        img, updated_color_mask, overlay_color, risk_level, flammability_percentage, water_percentage, poi, al_percentage,output_subfolder,alive_tree_pixels = process_image.process_image(img_path, relative_path)

        # Save results in the configured mode
        save_mask_or_visual(
            img, updated_color_mask, alive_tree_pixels, overlay_color, flammability_percentage,
            water_percentage, poi, al_percentage, risk_level, OUTPUT_MODE, output_subfolder, file_name
        )

print("Tile stitching and sphere conversion started...")
# Tile stitching and sphere conversion
cube_to_sphere(OUTPUT_FOLDER, level=TILE_LEVEL, krpano_tool_path=KRPANO_TOOL_PATH)
shutil.copy(os.path.join(OUTPUT_FOLDER, "merged_sphere_level_3.jpg"),  os.path.join(OUT, "merged_sphere_level_3.jpg"))

cube_to_sphere(OUTPUT_FOLDER_POI, level=TILE_LEVEL, krpano_tool_path=KRPANO_TOOL_PATH)
generate_hotspots_and_visualize(SPHERE_IMAGE_PATH, HOTSPOTS_XML_PATH)
shutil.copy(os.path.join(OUTPUT_FOLDER_POI, "hotspots.xml"),  os.path.join(OUT, "hotspots.xml"))

# update the folder path of marzipano
print("Updating Marzipano index...")
update_index_js(INPUT_FOLDER)


print("Processing completed successfully!")