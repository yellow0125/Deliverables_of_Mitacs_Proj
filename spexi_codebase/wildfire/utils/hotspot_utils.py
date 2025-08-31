import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

def normalize_contour(contour, width, height):
    return [(point[0][0] / width, point[0][1] / height) for point in contour]

# Function to create a Marzipano-compatible hotspot for a polygon
def create_marzipano_polygon_hotspot(polygon, name, fillcolor, bordercolor):
    hotspot = ET.Element("hotspot")
    hotspot.set("name", name)
    hotspot.set("keep", "false")
    hotspot.set("zorder", "0")
    hotspot.set("fillcolor", fillcolor)  # Dynamic fill color
    hotspot.set("fillalpha", "0.5")      # Example alpha
    hotspot.set("bordercolor", bordercolor)  # Dynamic border color
    hotspot.set("borderwidth", "3.0")       # Example border width
    hotspot.set("visible", "true")
    hotspot.set("enabled", "true")

    # Add points to the hotspot
    for x, y in polygon:
        point = ET.SubElement(hotspot, "point")
        ath = str(x * 360 - 180)  # Convert x from [0, 1] to [-180, 180]
        atv = str(-90 + y * 180)  # Convert y from [0, 1] to [-90, 90]
        point.set("ath", ath)
        point.set("atv", atv)

    return hotspot

# Function to convert polygons to Marzipano-compatible XML
def polygons_to_marzipano_xml(yellow_polygons, red_polygons, brown_polygons):
    root = ET.Element("hotspots")

    # Add yellow polygons with yellow color
    for i, polygon in enumerate(yellow_polygons):
        hotspot = create_marzipano_polygon_hotspot(polygon, f"yellow_{i+1}", "#FFFF00", "#FFFF00")
        root.append(hotspot)

    # Add red polygons with red color
    for i, polygon in enumerate(red_polygons):
        hotspot = create_marzipano_polygon_hotspot(polygon, f"red_{i+1}", "#FF0000", "#FF0000")
        root.append(hotspot)

    # Add brown polygons with brown color
    for i, polygon in enumerate(brown_polygons):
        hotspot = create_marzipano_polygon_hotspot(polygon, f"brown_{i+1}", "#8B4513", "#8B4513")
        root.append(hotspot)

    # Convert to string and pretty-print
    xml_str = ET.tostring(root, encoding="unicode")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

    return pretty_xml

# Function to generate hotspots XML and visualize the masks
def generate_hotspots_and_visualize(image_path, output_xml_path):
    
    image = cv2.imread(image_path)

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow, red, and brown masks in HSV
    yellow_lower = np.array([25, 40, 40])  # Example range for yellow
    yellow_upper = np.array([35, 255, 255])
    red_lower1 = np.array([0, 40, 40])    # Red has two ranges (low and high)
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 40, 40])
    red_upper2 = np.array([180, 255, 255])
    brown_lower = np.array([10, 40, 40])  # Example range for brown
    brown_upper = np.array([20, 255, 200])

    # Create masks for yellow, red, and brown
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # Combine red ranges
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

    # Find contours for yellow, red, and brown masks
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brown_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Image dimensions
    height, width = image.shape[:2]

    # Normalize contours to [0, 1] range
    yellow_polygons = [normalize_contour(cnt, width, height) for cnt in yellow_contours]
    red_polygons = [normalize_contour(cnt, width, height) for cnt in red_contours]
    brown_polygons = [normalize_contour(cnt, width, height) for cnt in brown_contours]

    # Combine polygons and generate the XML
    marzipano_xml = polygons_to_marzipano_xml(yellow_polygons, red_polygons, brown_polygons)

   # Save XML
    with open(output_xml_path, "w") as xml_file:
        xml_file.write(marzipano_xml)

    print(f"Hotspots XML saved to: {output_xml_path}")