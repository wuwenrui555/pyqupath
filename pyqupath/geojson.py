import json

import cv2
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from joblib import Parallel, delayed
from shapely.geometry import Polygon, mapping
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

################################################################################
# IO
################################################################################


def load_geojson_to_gdf(
    geojson_path: str = None,
    geojson_text: str = None,
) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON file or string as GeoPandas GeoDataFrame.

    Parameters
    ----------
    geojson_path : str, optional
        The file path to the GeoJSON file. If provided, the file will be read and parsed.
        This parameter is mutually exclusive with `geojson_text`.
    geojson_text : str, optional
        The GeoJSON string. If provided, the string will be parsed.
        This parameter is mutually exclusive with `geojson_path`.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoPandas GeoDataFrame containing the geometries and properties from the GeoJSON.
    """
    if geojson_path is not None:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
        # Convert GeoJSON to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    elif geojson_text is not None:
        geojson_data = json.loads(geojson_text)
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    else:
        raise ValueError("Either 'geojson_path' or 'geojson_text' must be provided.")
    return gdf


################################################################################
# Convert GeoJSON to mask
################################################################################


def polygon_to_mask(
    polygon: Polygon,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Generate a binary mask from a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon object defining the region of interest.
    shape : tuple
        The shape of the output mask as (height, width).

    Returns
    -------
    np.ndarray
        A binary mask with the same dimensions as the specified shape,
        where pixels inside the polygon are 1 and outside are 0.
    """
    height, width = shape

    # Rasterize the polygon
    mask = rasterize(
        [(polygon, 1)],  # Each tuple contains a geometry and the value to burn
        out_shape=(height, width),
        fill=0,  # Value for pixels outside the polygon
        dtype=np.uint8,
    )
    return mask


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
    color_dict : dict
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
    color_dict = {value: bright_colors[i] for i, value in enumerate(x)}

    return color_dict


def mask_to_geojson(
    mask: np.ndarray,
    geojson_path: str,
    annotation_dict: dict[int, str] = {},
    simplify_opencv_precision: float = None,
    simplify_shapely_tolerance: float = None,
):
    """
    Convert a labeled mask into a GeoJSON file.

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

    for label in tqdm(labels):
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


def binary_mask_to_polygon(binary_mask: np.ndarray) -> Polygon:
    """
    Convert a binary mask to a Shapely Polygon.

    Parameters
    ----------
    binary_mask : np.ndarray
        A 2D binary mask with values 0 or 1.

    Returns
    -------
    Polygon
        A Shapely Polygon representing the external contour of the binary mask.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        polygon = Polygon(contour.squeeze(axis=1))
    return polygon


def mask_to_polygon_batch(
    mask: np.ndarray,
    labels_batch: list[int],
) -> list[Polygon]:
    """
    Convert a batch of labels from a mask into Polygons.

    Parameters
    ----------
    mask : np.ndarray
        A 2D segmentation mask.
    labels_batch : list or np.ndarray
        A batch of unique label values from the mask.

    Returns
    -------
    list of Polygon
        A list of Shapely Polygons corresponding to the given labels.
    """
    return [
        binary_mask_to_polygon((mask == label).astype(np.uint8))
        for label in labels_batch
    ]


def mask_to_polygons(
    mask: np.ndarray,
    labels: list[int],
    batch_size: int = 10,
) -> list[Polygon]:
    """
    Convert a segmentation mask into a list of Polygons by processing labels in batches.

    Parameters
    ----------
    mask : np.ndarray
        A 2D segmentation mask.
    labels : list or np.ndarray
        A list of unique label values from the mask, excluding the background (e.g., 0).
    batch_size : int, optional
        The number of labels to process in each batch (default is 10).

    Returns
    -------
    list of Polygon
        A flattened list of Polygons corresponding to all labels in the mask.
    """
    # Split labels into batches
    labels_batches = [
        labels[i : i + batch_size] for i in range(0, len(labels), batch_size)
    ]

    # Process batches in parallel
    with tqdm_joblib(tqdm(desc="Processing", total=len(labels_batches), unit="batch")):
        polygons_batches = Parallel(n_jobs=-1)(
            delayed(mask_to_polygon_batch)(mask, labels_batch)
            for labels_batch in labels_batches
        )

    # Flatten the list of batches into a single list
    polygons = [
        polygon for polygons_batch in polygons_batches for polygon in polygons_batch
    ]
    return polygons


def mask_to_geojson_joblib(
    mask: np.ndarray,
    geojson_path: str,
    annotation_dict: dict[int, str] = {},
):
    """
    Convert a labeled mask into a GeoJSON file using parallel processing.

    Parameters
    ----------
    mask : np.ndarray
        A 2D segmentation mask where each unique value represents a labeled region.
    geojson_path : str
        The file path to save the resulting GeoJSON file.
    annotation_dict : dict[int, str], optional
        A dictionary mapping integer label values in the mask to their annotation names.
        Labels not in this dictionary will be classified as "Unknown". Default is an empty dictionary.

    Returns
    -------
    None
    """
    # Extract unique labels from the mask
    labels = np.unique(mask)
    labels = labels[labels != 0]

    # Convert mask to polygons
    polygons = mask_to_polygons(mask, labels)

    # Assign bright colors to annotations
    color_dict = assign_bright_colors(np.unique(list(annotation_dict.values())))
    color_dict["Unknown"] = (128, 128, 128)  # Gray for unknown annotations

    # Create GeoJSON features
    features = [
        {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "objectType": "annotation",
                "name": str(label),
                "classification": {
                    "name": annotation_dict.get(label, "Unknown"),
                    "color": color_dict[annotation_dict.get(label, "Unknown")],
                },
            },
        }
        for polygon, label in zip(polygons, labels)
    ]

    # Create GeoJSON object
    geojson = {"type": "FeatureCollection", "features": features}
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)
