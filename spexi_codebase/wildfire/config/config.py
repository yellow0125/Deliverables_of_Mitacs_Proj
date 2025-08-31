
import numpy as np




# Directories
INPUT_FOLDER = '7b947867-7ab5-435e-9cdb-a7306f71bdbe'  # Folder containing input images
OUTPUT_FOLDER = 'result/'  # Output folder for segmented masks
OUTPUT_FOLDER_POI = 'result_poi/'  # Output folder for points of interest
OUT = 'output'  # output folder for all data




# Model paths
MODEL_CONFIG = 'models/config_output.py'
MODEL_CHECKPOINT = 'models/best_mIoU_iter_2000.pth'
DEVICE = 'cpu'  # Change to 'cuda' if using a GPU

WATER_MODEL_PATH = 'models/water_model.pth'
WATER_MODEL_CONFIG = 'models/config_water.py'



# Tile stitching and conversion
TILE_LEVEL = 3
KRPANO_TOOL_PATH = "<path_to_installed_krpano>/krpano-1.22.2/krpanotools"

# Classes and color palette
CLASSES = ('bg', 'Beetle-fire tree', 'Dead tree', 'Debris', 'Water', 'Alive Tree')
PALETTE = np.array([
    [0, 0, 0],        # Background
    [0, 0, 255],      # Beetle-fire tree
    [0, 255, 255],    # Dead tree
    [19, 69, 139],    # Debris
    [255, 0, 0],      # Water
    [0, 255, 0]       # Alive Tree
])

# Risk levels
RISK_LEVELS = {
    "Low": {"range": (0, 5.99), "color": [0, 255, 0]},
    "Moderate": {"range": (6, 25.99), "color": [0, 255, 255]},
    "High": {"range": (26, 60.99), "color": [0, 165, 255]},
    "Very High": {"range": (61, 80.99), "color": [0, 0, 255]},
    "Extreme": {"range": (81, float('inf')), "color": [0, 0, 139]}
}

# Other parameters
OUTPUT_MODE = 'mask'  # 'mask', 'segmented', or 'visual'.Use 'segmented' or 'visual' for debugging purposes only
CONTRAST_THRESHOLD = 70
HIGHLIGHT_AMOUNT = 2
PERCENTILE = 50


# Paths for merged sphere image
SPHERE_IMAGE_PATH = 'result_poi/merged_sphere_level_3.jpg'
HOTSPOTS_XML_PATH = 'result_poi/hotspots.xml'
