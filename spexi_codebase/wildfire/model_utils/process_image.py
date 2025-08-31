import os
import cv2
import mmcv
import numpy as np
from mmseg.apis import init_model,inference_model
from utils.file_utils import save_color_mask, create_output_dirs
from utils.image_utils import is_high_contrast, increase_highlights_auto_threshold, detect_green_areas
from utils.segmentation_utils import calculate_pixels_per_color, assign_risk_level, subtract_overlapping_pixels
from config.config import (
    MODEL_CONFIG, MODEL_CHECKPOINT, INPUT_FOLDER, OUTPUT_FOLDER, OUTPUT_FOLDER_POI,
    TILE_LEVEL, KRPANO_TOOL_PATH,
    WATER_MODEL_PATH, SPHERE_IMAGE_PATH, HOTSPOTS_XML_PATH,  # Added paths
    CLASSES, DEVICE, PALETTE, RISK_LEVELS, OUTPUT_MODE, CONTRAST_THRESHOLD, HIGHLIGHT_AMOUNT, PERCENTILE,WATER_MODEL_CONFIG
)

class ProcessImage:
    def __init__(self):
        """
        Initialize the ProcessImage class.

        Args:
            model: The segmentation model.
            water_model: The water segmentation model.
            config: Config module containing required parameters.
        """
        self.output_folder = os.path.join(OUTPUT_FOLDER, str(TILE_LEVEL))
        self.output_folder_poi = os.path.join(OUTPUT_FOLDER_POI, str(TILE_LEVEL))
        if DEVICE == 'cpu':
             self.model = init_model(MODEL_CONFIG, MODEL_CHECKPOINT,device='cpu')
        else:
             self.model = init_model(MODEL_CONFIG, MODEL_CHECKPOINT,device='cuda:0')
        
        if DEVICE == 'cpu':
             self.water_model = init_model(WATER_MODEL_CONFIG, WATER_MODEL_PATH,device='cpu')
        else:
             self.water_model = init_model(WATER_MODEL_CONFIG, WATER_MODEL_PATH,device='cuda:0')

        
        # self.water_model = load_water_segmentation_model(WATER_MODEL_PATH)
        create_output_dirs(self.output_folder, self.output_folder_poi)
        
    
    def segment_image(self, img):
        
        """
        Perform segmentation on the image and generate the color mask.
        """
        result = inference_model(self.model, img)

        # Extract the segmentation map and convert it into a color mask
        seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()
        color_mask = PALETTE[seg_map]
        return color_mask
    
    def preprocess_image_alive_tree(self, img,img_path):
        """
        Preprocess the image based on contrast.
        """
        if is_high_contrast(img, CONTRAST_THRESHOLD):
            print(f"High contrast detected for {os.path.basename(img_path)}, applying highlight enhancement.")
            processed_image = increase_highlights_auto_threshold(img, HIGHLIGHT_AMOUNT, PERCENTILE)
        else:
            # print(f"No high contrast detected for {os.path.basename(img_path)}, using the original image.")
            processed_image = img
        
        return img
        
    
    def detect_water_areas(self, img, color_mask):
        
        """
            Detect water areas using the water segmentation model and update the color mask.

            Args:
                img: The input image as a numpy array.
                color_mask: The current color mask from segmentation.

            Returns:
                Updated color mask with water areas marked.
        """
        # Run inference using the water segmentation model
        water_result = inference_model(self.water_model, img)

        # Extract the segmentation map
        seg_map = water_result.pred_sem_seg.data.cpu().numpy().squeeze()

        # Get the index for the 'Water' class
        water_class_index = CLASSES.index('Water')

        # Create a binary mask for the 'Water' class
        water_mask = (seg_map == water_class_index).astype(np.uint8)

        # Update the color mask to mark water areas (e.g., red)
        color_mask[water_mask > 0] = [255, 0, 0]

        return color_mask

    

    def process_image(self, img_path, relative_path):
        """
        Process a single image: segmentation, water detection, and flammability analysis.

        Args:
            img_path: Path to the input image.
            relative_path: Relative path for organizing output directories.

        Returns:
            Tuple of original image, updated color mask, overlay color, and risk level.
        """
        # Load the image
        img = mmcv.imread(img_path)
        color_mask = self.segment_image(img)
     
        
         # Save the color mask
        save_color_mask(color_mask, self.output_folder_poi, relative_path, os.path.basename(img_path))
        color_mask = self.detect_water_areas(img, color_mask)
        processed_image = self.preprocess_image_alive_tree(img,img_path)
        green_mask = detect_green_areas(processed_image)
        color_mask[green_mask > 0] = [0, 255, 0]
        
        # Subtract overlapping pixels for the target class 'Alive Tree'
        updated_color_mask = subtract_overlapping_pixels(color_mask, PALETTE, 'Alive Tree', CLASSES)

        # Calculate pixel counts for analysis
        pixel_counts = calculate_pixels_per_color(updated_color_mask, PALETTE)
        total_pixels = updated_color_mask.shape[0] * updated_color_mask.shape[1]
        
        water_pixels = pixel_counts.get(tuple([255, 0, 0]), 0)
        alive_tree_pixels = pixel_counts.get(tuple([0, 255, 0]), 0)

        water_percentage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        al_percentage = (alive_tree_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Further processing for risk level analysis
        flammability_percentage = 0
        poi = 0
        for color, count in pixel_counts.items():
            class_name = CLASSES[PALETTE.tolist().index(list(color))]
            if class_name in ["Dead tree", "Debris", "Beetle-fire tree"]:
                if alive_tree_pixels > 0:
                    percentage = (count / alive_tree_pixels) * 100
                else:
                    # Divide by total_pixels if alive_tree_pixels is zero
                    percentage = (count / 1) * 100 if total_pixels > 0 else 0
                poi += percentage

        if poi > 0:
            flammability_percentage = poi * (1 - water_percentage / 100)

        overlay_color, risk_level = assign_risk_level(flammability_percentage, RISK_LEVELS)
     
        output_subfolder = os.path.join( self.output_folder, relative_path)
        create_output_dirs(output_subfolder)
        
        

        return (
        img, updated_color_mask, overlay_color, risk_level,
        flammability_percentage, water_percentage, poi, al_percentage,output_subfolder,alive_tree_pixels
    )
