import json

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping

################################################################################
# Convert segmentation mask to GeoJSON
################################################################################


def assign_bright_colors(x: list[str]) -> dict[str, tuple[int, int, int]]:
    """
    Assign bright RGB colors to variable values.

    Parameters
    ----------
    x : list or array
        A list or array of unique variable values to assign colors to.

    Returns
    -------
    color_map : dict
        A dictionary mapping each variable value to a bright RGB color.
    """
    # Define a list of bright colors in RGB
    bright_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Sky Blue
        (255, 255, 255),  # White
    ]

    # Ensure there are enough colors
    if len(x) > len(bright_colors):
        np.random.seed(1)  # For reproducibility
        # Randomly generate bright colors if there aren't enough predefined
        for _ in range(len(x) - len(bright_colors)):
            bright_colors.append(tuple(np.random.choice(range(128, 256), 3)))

    # Map each variable value to a color
    color_map = {value: bright_colors[i] for i, value in enumerate(x)}

    return color_map


def mask_to_geojson(
    mask: np.ndarray,
    geojson_path: str,
    annotation_dict: dict[int, str] = {},
    simplify_opencv_precision: float = None,
    simplify_shapely_tolerance: float = None,
):
    """
    Convert a labeled segmentation mask into a compact GeoJSON file.

    Parameters
    ----------
    mask : np.ndarray
        A 2D array where values represent labels for different segmented regions.
    geojson_path : str
        Path to save the output GeoJSON file.
    annotation_dict : dict[int, str]
        A dictionary mapping label integers to their corresponding annotation names.
    simplify_opencv_precision : float, optional
        Precision value for OpenCV's approxPolyDP. Smaller values retain more detail.
        Default is None, meaning no OpenCV simplification is applied. (Recommended: 0.01)
    simplify_shapely_tolerance : float, optional
        Tolerance for simplifying polygons using Shapely. Smaller values retain
        more detail. Default is None, meaning no simplification is applied.

    Returns
    -------
    None
    """
    labels = np.unique(mask)
    features = []

    # Assign colors to annotations
    color_map = assign_bright_colors(np.unique(list(annotation_dict.values())))
    color_map["Unknown"] = (128, 128, 128)  # Gray for unknown annotations

    for label in labels:
        if label == 0:  # Skip background
            continue

        annotation = annotation_dict.get(label, "Unknown")
        color = color_map[annotation]

        # Create a binary mask for the current label
        binary_mask = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if simplify_opencv_precision is not None:
                # Simplify the contour using OpenCV's approxPolyDP
                epsilon = simplify_opencv_precision * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                polygon = Polygon(approx.squeeze(axis=1))
            else:
                # Use the original contour without simplification
                polygon = Polygon(contour.squeeze(axis=1))

            if simplify_shapely_tolerance is not None:
                polygon = polygon.simplify(simplify_shapely_tolerance)

            if polygon.is_valid and not polygon.is_empty:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(polygon),
                        "properties": {
                            "objectType": "annotation",
                            "name": str(label),
                            "classification": {
                                "name": annotation_dict.get(label, "Unkonwn"),
                                "color": color,
                            },
                        },
                    }
                )

    geojson = {"type": "FeatureCollection", "features": features}

    # Save GeoJSON file with compact formatting
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)
    print(f"GeoJSON saved to {geojson_path}")
