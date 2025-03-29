# %%
import json
from collections import OrderedDict
from pathlib import Path
from typing import Generator, Optional, Union

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from joblib import delayed
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, Polygon, mapping
from tqdm import tqdm
from tqdm_joblib import ParallelPbar

from pyqupath import constants
from pyqupath.color import assign_bright_colors

################################################################################
# GeoJSON IO
################################################################################


class GeojsonProcessor:
    """
    A class for reading and manipulating GeoJSON files.

    Attributes:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the GeoJSON data.
    """

    DEFAULT_CLASSIFICATION = json.dumps({"name": "unknown", "color": [128, 128, 128]})

    def __init__(self, gdf: gpd.GeoDataFrame):
        """
        Initialize a GeojsonProcessor with a GeoDataFrame.
        """
        # Set name as string
        gdf["name"] = gdf["name"].astype(str)

        # Set default classification if not present
        if "classification" not in gdf.columns:
            gdf["classification"] = GeojsonProcessor.DEFAULT_CLASSIFICATION

        # Store raw GeoDataFrame
        self.gdf_raw = gdf.copy()

        # Set index
        self.gdf = gdf
        self.set_index(index="name", inplace=True)

    @classmethod
    def from_path(cls, geojson_f: Union[Path, str]):
        """
        Create a GeojsonProcessor from a GeoJSON file path.
        """
        gdf = gpd.read_file(geojson_f)
        return cls(gdf)

    @classmethod
    def from_text(cls, geojson_text: str):
        """
        Create a GeojsonProcessor from a GeoJSON text string.
        """
        geojson_data = json.loads(geojson_text)
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
        return cls(gdf)

    @staticmethod
    def _add_classification(gdf: gpd.GeoDataFrame):
        """
        Add classification information to the GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to add classification information to.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame with classification information added.
        """
        cls_data = []
        for item in gdf["classification"]:
            item = json.loads(item)
            name = item.get("name", "")
            color_rgb = item.get("color", "")
            color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
            cls_data.append({"cls_name": name, "cls_color": color_hex})
        cls_df = pd.DataFrame(cls_data)
        cls_df.index = gdf.index
        gdf = pd.concat([gdf, cls_df], axis=1)
        return gdf

    @staticmethod
    def _plot_classification(
        gdf: gpd.GeoDataFrame,
        figsize: tuple = (10, 10),
        legend: bool = True,
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """
        Plot the classification of the GeoDataFrame.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure. Default is (10, 10).
        legend : bool, optional
            Whether to show the legend. Default is True.
        ax : plt.Axes, optional
            The axis to plot on. Default is None.

        Returns
        -------
        plt.Figure
            The figure object.
        """
        gdf = GeojsonProcessor._add_classification(gdf)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        gdf.plot(
            ax=ax,
            legend=False,
            color=gdf["cls_color"],
            aspect=1,  # Set a valid aspect ratio
        )
        ax.invert_yaxis()

        if legend:
            unique_classes = gdf[["cls_name", "cls_color"]].drop_duplicates()
            for _, (cls_name, cls_color) in unique_classes.iterrows():
                ax.scatter([], [], c=cls_color, label=cls_name)
            ax.legend(
                title="Classification",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

        ax.set_aspect("equal")

        return fig

    def set_index(
        self, index: str = "name", inplace: bool = False
    ) -> Union[None, gpd.GeoDataFrame]:
        """
        Set the index of the GeoDataFrame and handle duplicates by adding numeric suffixes.

        Parameters
        ----------
        index : str, default="name"
            The column name to use as index.
        inplace : bool, optional
            Whether to modify the GeoDataFrame in place. Default is False.

        Returns
        -------
        gpd.GeoDataFrame
            The modified GeoDataFrame.
        """
        # Add suffix to duplicate names
        cumcount_name = self.gdf.groupby(index).cumcount() + 1
        n_index = self.gdf[index].value_counts()
        duplicates = n_index[n_index > 1].index.tolist()
        new_indices = [
            f"{name}" if name not in duplicates else f"{name}_{cumcount_name}"
            for name, cumcount_name in zip(self.gdf[index], cumcount_name)
        ]

        # Print the duplicates
        if len(duplicates) > 0:
            print(f"Duplicate values found:\n{n_index[n_index > 1]}")

        # Set the new index
        if inplace:
            self.gdf.index = new_indices
        else:
            gdf = self.gdf.copy()
            gdf.index = new_indices
            return gdf

    def plot_classification(
        self,
        figsize: tuple = (10, 10),
        legend: bool = True,
        plot_raw: bool = False,
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """
        Plot the classification of the GeoDataFrame.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure. Default is (10, 10).
        legend : bool, optional
            Whether to show the legend. Default is True.
        plot_raw : bool, optional
            Whether to plot the raw GeoDataFrame. Default is False.
        ax : plt.Axes, optional
            The axis to plot on. Default is None.

        Returns
        -------
        plt.Figure
            The figure object.
        """
        if plot_raw:
            gdf = self.gdf_raw
        else:
            gdf = self.gdf
        return self._plot_classification(gdf, figsize, legend, ax)

    def update_classification(
        self,
        name_dict: dict[Union[str, int], Union[str, int]],
        color_dict: Optional[dict[Union[str, int], tuple[int, int, int]]] = None,
    ) -> None:
        """
        Update classification names and colors in the GeoDataFrame.

        Parameters
        ----------
        name_dict : dict[Union[str, int], Union[str, int]]
            Dictionary mapping original names to new classification names.
        color_dict : Optional[dict[Union[str, int], tuple[int, int, int]]], optional
            Dictionary mapping classification names to RGB colors.
            If not provided, colors will be automatically assigned.
        """

        # Generate colors if not provided
        if color_dict is None:
            unique_names = list(set(name_dict.values()))
            color_dict = assign_bright_colors(unique_names)

        # Update classification
        cls_name = self.gdf["name"].map(name_dict)
        cls_color = cls_name.map(color_dict)
        self.gdf["classification"] = [
            json.dumps({"name": name, "color": color})
            for name, color in zip(cls_name, cls_color)
        ]

    def output_geojson(self, output_f: Path):
        """
        Output the GeoDataFrame as a GeoJSON file.
        """
        self.gdf.to_file(output_f, driver="GeoJSON")


################################################################################
# Polygon Processing
################################################################################


class PolygonProcessor:
    """
    A class for processing polygon geometries and applying them to images.
    """

    def __init__(self, polygon: Polygon):
        """
        Initialize a PolygonProcessor with a Shapely polygon.
        """
        self.polygon = polygon

    @staticmethod
    def polygon_to_mask(polygon: Polygon, shape: tuple[int, int]) -> np.ndarray:
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
            where pixels inside the polygon are True and outside are False.
        """
        # Rasterize the polygon
        mask = rasterize(
            [(polygon, True)],  # Set True within the polygon
            out_shape=shape,
            fill=False,  # Set False outside the polygon
            dtype=np.uint8,
        ).astype(bool)
        return mask

    def crop_array_by_polygon(
        self,
        img: Union[np.ndarray, zarr.Array],
        dim_order: str = "CYX",
        fill_value: float = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop an image using a polygon and apply a mask.

        Parameters
        ----------
        img : np.ndarray or zarr.Array
            The image to crop.
        dim_order : str, optional
            The dimension order of the image. Default is "CYX".
            Supported values are "CYX" (channel, y, x) and "YXC" (y, x, channel).
        fill_value : float, optional
            Value to fill outside the polygon. Default is 0.

        Returns
        -------
        tuple
            A tuple containing:
            - masked_image: The cropped image with mask applied
            - mask: The binary mask

        Raises
        ------
        ValueError
            If the image dimension order is not supported or if the image is not 2D or 3D.
        """
        # Get image dimensions from the image
        height, width = img.shape[-2:]

        # Calculate bounds of the polygon
        y_min, x_min, y_max, x_max = self.polygon.bounds
        y_min = max(0, int(np.floor(y_min)))
        y_max = min(height, int(np.ceil(y_max)))
        x_min = max(0, int(np.floor(x_min)))
        x_max = min(width, int(np.ceil(x_max)))

        # Shift the polygon to the cropped region
        y_coords = [y - y_min for y in self.polygon.exterior.coords.xy[0]]
        x_coords = [x - x_min for x in self.polygon.exterior.coords.xy[1]]
        shifted_polygon = Polygon(zip(x_coords, y_coords))

        # Create mask for the shifted polygon
        shifted_mask = PolygonProcessor.polygon_to_mask(
            shifted_polygon, (y_max - y_min, x_max - x_min)
        )

        # Crop the image based on dimension order
        if img.ndim == 2:
            shifted_img = img[y_min:y_max, x_min:x_max]
            mask = shifted_mask
        elif img.ndim == 3:
            if dim_order == "CYX":
                shifted_img = img[:, y_min:y_max, x_min:x_max]
                mask = shifted_mask[np.newaxis, :, :]
            elif dim_order == "YXC":
                shifted_img = img[y_min:y_max, x_min:x_max, :]
                mask = shifted_mask[:, :, np.newaxis]
            else:
                raise ValueError(
                    f"Unsupported dimension order: {dim_order}. Use 'CYX' or 'YXC'."
                )
        else:
            raise ValueError(f"Image must be 2D or 3D, got {img.ndim}D")

        # Apply mask and fill values
        img_masked = shifted_img * mask + int(fill_value) * (1 - mask)

        return img_masked, mask


def crop_dict_by_geojson_batch(
    img_dict: dict[str, Union[np.ndarray, zarr.Array]],
    geojson_f: Union[Path, str],
    fill_value: float = 0,
) -> Generator[tuple[str, dict[str, np.ndarray]], None, None]:
    """
    Crop a dictionary of images (2D numpy arrays) by a list of polygons.

    Parameters
    ----------
    img_dict : dict[str, Union[np.ndarray, zarr.Array]]
        A dictionary of images to crop.
    geojson_f : Union[Path, str]
        The path to the geojson file.
    fill_value : float, optional
        Value to fill outside the polygon. Default is 0.

    Returns
    -------
    Generator[tuple[str, dict[str, np.ndarray]], None, None]
        A generator of tuples containing:
        - name: The name of the polygon
        - cropped_img_dict: A dictionary of cropped images with mask applied
    """
    # Validate the image dimensions
    if not all(img.ndim == 2 for img in img_dict.values()):
        raise ValueError("All images in img_dict must be 2D numpy arrays")

    # Read the geojson file
    geojson_reader = GeojsonProcessor.from_path(geojson_f)
    names = geojson_reader.gdf.index.tolist()
    polygons = geojson_reader.gdf["geometry"].tolist()

    # Crop the images
    for name, polygon in zip(names, polygons):
        polygon_processor = PolygonProcessor(polygon)
        cropped_img_dict = {
            ch: polygon_processor.crop_array_by_polygon(
                img_dict[ch], fill_value=fill_value
            )[0]
            for ch in img_dict.keys()
        }
        yield name, cropped_img_dict


def crop_array_by_geojson_batch(
    img: Union[np.ndarray, zarr.Array],
    geojson_f: Union[Path, str],
    dim_order: str = "CYX",
    fill_value: float = 0,
) -> Generator[tuple[str, np.ndarray], None, None]:
    """
    Crop an image (2D or 3D numpy array) using a list of polygons.

    Parameters
    ----------
    img : np.ndarray or zarr.Array
        The image to crop.
    geojson_f : Union[Path, str]
        The path to the geojson file.
    dim_order : str, optional
        The dimension order of the image. Default is "CYX".
        Supported values are "CYX" (channel, y, x) and "YXC" (y, x, channel).
    fill_value : float, optional
        Value to fill outside the polygon. Default is 0.

    Returns
    -------
    Generator[tuple[str, np.ndarray], None, None]
        A generator of tuples containing:
        - name: The name of the polygon
        - cropped_img: The cropped image with mask applied
    """
    # Validate the image dimensions
    if img.ndim not in [2, 3]:
        raise ValueError("Image must be 2D or 3D numpy array")

    # Validate the dimension order
    if dim_order not in ["CYX", "YXC"]:
        raise ValueError("dim_order must be 'CYX' or 'YXC'")

    # Read the geojson file
    geojson_reader = GeojsonProcessor.from_path(geojson_f)
    names = geojson_reader.gdf.index.tolist()
    polygons = geojson_reader.gdf["geometry"].tolist()

    # Crop the image
    for name, polygon in zip(names, polygons):
        polygon_processor = PolygonProcessor(polygon)
        cropped_img, _ = polygon_processor.crop_array_by_polygon(
            img, dim_order, fill_value
        )
        yield name, cropped_img


################################################################################
# Convert segmentation mask to GeoJSON
################################################################################


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
        A dictionary mapping label integers to their corresponding annotation
        names.
    simplify_opencv_precision : float, optional
        Precision value for OpenCV's approxPolyDP. Smaller values retain more
        detail. Default is None, meaning no OpenCV simplification is applied.
        (Recommended: 0.01)
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

    for label in tqdm(labels, bar_format=constants.TQDM_FORMAT):
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
            # Skip contours with insufficient points
            if len(contour) < 4:
                print(f"Skipping contour with insufficient points: {label}, {contour}")
                continue

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
                                "name": annotation_dict.get(label, "Unknown"),
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


def binary_mask_to_polygon(
    binary_mask: np.ndarray,
    diagonal: bool = False,
) -> Polygon:
    """
    Convert a binary mask to a Shapely Polygon.

    Parameters
    ----------
    binary_mask : np.ndarray
        A 2D binary mask with values 0 or 1.
    diagonal : bool, optional
        Whether to allow diagonal lines in the polygon. Default if False.  If
        False, adjusts the contour to be either vertical or horizontal.

    Returns
    -------
    Polygon
        A Shapely Polygon representing the external contour of the binary mask.
    """

    def _get_diagonal_points(curr_x, curr_y, next_x, next_y):
        """
        Get all diagonal points between two diagonal points.

        Parameters
        ----------
        curr_x : int
            X-coordinate of current point.
        curr_y : int
            Y-coordinate of current point.
        next_x : int
            X-coordinate of next point.
        next_y : int
            Y-coordinate of next point.

        Returns
        -------
        list
            A list of diagonal points between the current and next points.
        """
        points = []
        dx = 1 if next_x > curr_x else -1
        dy = 1 if next_y > curr_y else -1

        x, y = curr_x, curr_y
        while x != next_x or y != next_y:
            points.append((x, y))
            x += dx
            y += dy
        points.append((next_x, next_y))
        return points

    def _adjust_to_axis(contour, geojson_loc):
        """
        Adjusts a contour to be either vertical or horizontal.

        Parameters
        ----------
        contour : np.ndarray
            Contour points as a numpy array of shape (N, 2).
        geojson_loc : np.ndarray
            Location of the mask in geojson format.

        Returns
        -------
        list
            Adjusted contour points with only vertical or horizontal lines.
        """
        adjusted_contour = []
        n = len(contour)

        for i in range(n):
            # Current and next point
            curr_point = contour[i]
            curr_x, curr_y = curr_point
            next_point = contour[(i + 1) % n]  # Wrap around for closed contour
            next_x, next_y = next_point

            # Add the current point
            adjusted_contour.append(curr_point)

            # If the line is diagonal
            if curr_x != next_x and curr_y != next_y:
                # Get all diagonal points
                diagonal_points = _get_diagonal_points(curr_x, curr_y, next_x, next_y)
                n_diagonal = len(diagonal_points)

                # Add intermediate points if they are in the mask
                for j in range(n_diagonal - 1):
                    curr_x, curr_y = diagonal_points[j]
                    next_x, next_y = diagonal_points[j + 1]
                    if geojson_loc[curr_y, next_x] != 0:
                        intermediate_point = [next_x, curr_y]
                        adjusted_contour.append(intermediate_point)
                    elif geojson_loc[next_y, curr_x] != 0:
                        intermediate_point = [curr_x, next_y]
                        adjusted_contour.append(intermediate_point)
                    adjusted_contour.append([next_x, next_y])

        return np.array(adjusted_contour)

    # location of the mask in geojson format
    shape_0 = binary_mask.shape[0]
    shape_1 = binary_mask.shape[1]
    geojson_loc = np.zeros((shape_0 + 1, shape_1 + 1), dtype=np.uint8)
    geojson_loc[:shape_0, :shape_1] += binary_mask
    geojson_loc[:shape_0, 1:] += binary_mask
    geojson_loc[1:, :shape_1] += binary_mask
    geojson_loc[1:, 1:] += binary_mask

    contours, _ = cv2.findContours(
        geojson_loc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if diagonal:
        for contour in contours:
            polygon = Polygon(contour.squeeze(axis=1))
    else:
        for contour in contours:
            adjusted_contour = _adjust_to_axis(contour.squeeze(axis=1), geojson_loc)
            polygon = Polygon(adjusted_contour)
    return polygon


def mask_to_polygon_batch(
    mask: np.ndarray,
    labels_batch: list[int],
    diagonal: bool = False,
) -> list[Polygon]:
    """
    Convert a batch of labels from a mask into Polygons.

    Parameters
    ----------
    mask : np.ndarray
        A 2D segmentation mask.
    labels_batch : list or np.ndarray
        A batch of unique label values from the mask.
    diagonal : bool, optional
        Whether to allow diagonal lines in the polygon. Default if False. If
        False, adjusts the contour to be either vertical or horizontal.

    Returns
    -------
    list of Polygon
        A list of Shapely Polygons corresponding to the given labels.
    """
    return [
        binary_mask_to_polygon((mask == label).astype(np.uint8), diagonal)
        for label in labels_batch
    ]


def mask_to_polygons(
    mask: np.ndarray,
    labels: list[int],
    n_jobs: int = 10,
    batch_size: int = 10,
    diagonal: bool = False,
) -> list[Polygon]:
    """
    Convert a segmentation mask into a list of Polygons by processing labels in
    batches.

    Parameters
    ----------
    mask : np.ndarray
        A 2D segmentation mask.
    labels : list or np.ndarray
        A list of unique label values from the mask, excluding the background
        (e.g., 0).
    n_jobs : int, optional
        The number of parallel workers (CPU cores or threads) are spawned to
        process the tasks. Default is 10.
    batch_size : int, optional
        The number of labels to process in each batch (default is 10).
    diagonal : bool, optional
        Whether to allow diagonal lines in the polygon. Default if False. If
        False, adjusts the contour to be either vertical or horizontal.

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
    polygons_batches = ParallelPbar(
        desc="Mask to polygons",
        bar_format=constants.TQDM_FORMAT,
    )(n_jobs=n_jobs)(
        delayed(mask_to_polygon_batch)(mask, labels_batch, diagonal)
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
    n_jobs: int = 10,
    batch_size: int = 10,
    diagonal: bool = False,
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
        A dictionary mapping integer label values in the mask to their
        annotation names. Labels not in this dictionary will be classified as
        "Unknown". Default is an empty dictionary.
    n_jobs : int, optional
        The number of parallel workers (CPU cores or threads) are spawned to
        process the tasks. Default is 10.
    batch_size : int, optional
        The number of labels to process in each batch (default is 10).
    diagonal : bool, optional
        Whether to allow diagonal lines in the polygon. Default if False. If
        False, adjusts the contour to be either vertical or horizontal.

    Returns
    -------
    None
    """
    # Extract unique labels from the mask
    labels = np.unique(mask)
    labels = labels[labels != 0]

    # Convert mask to polygons
    polygons = mask_to_polygons(mask, labels, n_jobs, batch_size, diagonal)

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


################################################################################
# Deprecated
################################################################################


def load_geojson_to_gdf(
    geojson_path: str = None,
    geojson_text: str = None,
) -> gpd.GeoDataFrame:
    """
    (Deprecated) Load a GeoJSON file or string as GeoPandas GeoDataFrame.

    Parameters
    ----------
    geojson_path : str, optional
        The file path to the GeoJSON file. If provided, the file will be read
        and parsed. This parameter is mutually exclusive with `geojson_text`.
    geojson_text : str, optional
        The GeoJSON string. If provided, the string will be parsed.
        This parameter is mutually exclusive with `geojson_path`.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoPandas GeoDataFrame containing the geometries and properties from
        the GeoJSON.
    """
    if geojson_path is not None:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    elif geojson_text is not None:
        geojson_data = json.loads(geojson_text)
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    else:
        raise ValueError("Either 'geojson_path' or 'geojson_text' must be provided.")
    return gdf


def update_geojson_classification(
    geojson_f: Union[Path, str],
    output_f: Union[Path, str],
    name_dict: dict[Union[str, int], Union[str, int]],
    color_dict: Optional[dict[Union[str, int], tuple[int, int, int]]] = None,
) -> None:
    """
    Update classification names and colors in a GeoJSON file.

    Parameters
    ----------
    geojson_f : Union[Path, str]
        Input GeoJSON file path
    output_f : Union[Path, str]
        Output GeoJSON file path
    name_dict : Union[dict[str, str], dict[int, str]]
        Dictionary mapping original names to new classification names
    color_dict : Optional[dict[Union[str, int], tuple[int, int, int]]], optional
        Dictionary mapping classification names to RGB colors.
        If not provided, colors will be automatically assigned.
    """
    # Read input GeoJSON
    with open(geojson_f, "r") as f:
        geojson_data = json.load(f)

    # Generate colors if not provided
    if color_dict is None:
        unique_names = list(set(name_dict.values()))
        color_dict = assign_bright_colors(unique_names)

    # Update features
    features = geojson_data["features"]
    for feature in features:
        properties = feature["properties"]
        if "name" in properties:
            orig_name = properties["name"]
            if orig_name in name_dict:
                new_name = name_dict[orig_name]
                properties["classification"] = {
                    "name": new_name,
                    "color": color_dict[new_name],
                }

    # Write output
    output_path = Path(output_f)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson_data, f)


def crop_dict_by_geojson(
    im_dict: OrderedDict[str, np.ndarray],
    path_geojson: str,
    fill: float = 0,
    multipolygon_rate: float = 100,
    desc: str = "Cropping",
) -> Generator[tuple[str, OrderedDict[str, np.ndarray]], None, None]:
    """
    Crop regions from images in a dictionary using polygons from a GeoJSON file.

    This function takes an ordered dictionary of images and a GeoJSON file path.
    For each polygon in the GeoJSON, it crops the region specified by the polygon
    and applies a mask to keep only the pixels inside the polygon, while filling
    the outside region with a specified value.

    If the geometry in the GeoJSON is a MultiPolygon, the function will attempt
    to determine the main Polygon within the MultiPolygon by comparing the areas
    of the polygons. If the area of the largest polygon is significantly larger
    than the second largest polygon (determined by the `multipolygon_rate`), the
    largest polygon will be used. Otherwise, the region will be skipped.

    Parameters
    ----------
    im_dict : OrderedDict[str, np.ndarray]
        A dictionary where keys are channel names and values are 2D NumPy arrays
        representing image channels.
    path_geojson : str
        Path to the GeoJSON file containing the polygons for cropping.
    fill : float, optional
        Value to fill in the masked areas outside the polygon. Default is 0.
    multipolygon_rate : float, optional
        The ratio of the area of the largest polygon to the second largest polygon
        within a MultiPolygon. If the ratio is greater than this value, the largest
        polygon will be used. Default is 100.
    desc : str, optional
        Description for the progress bar. Default is "Cropping".

    Yields
    ------
    tuple[str, OrderedDict[str, np.ndarray]]
        A tuple containing:
        - The name of the region from the GeoJSON
        - An ordered dictionary with the cropped and masked regions for each channel
    """
    # Get the shape of the first image (all images should have same shape)
    im_shape = next(iter(im_dict.values())).shape

    # Load GeoJSON into GeoDataFrame
    gdf = load_geojson_to_gdf(path_geojson)

    # Setup progress bar
    n_regions = len(gdf)
    n_channels = len(im_dict)
    pb = tqdm(
        total=n_regions * n_channels,
        desc=desc,
        bar_format=constants.TQDM_FORMAT,
    )

    for _, row in gdf.iterrows():
        geometry, name = row[["geometry", "name"]]

        # Handle different geometry types
        if isinstance(geometry, Polygon):
            polygon = geometry
        elif isinstance(geometry, MultiPolygon):
            # Try to extract main polygon from MultiPolygon
            polygons = [p for p in geometry.geoms if isinstance(p, Polygon)]
            if not polygons:
                print(f"Skipping {name}: MultiPolygon contains no valid Polygons")
                continue

            areas = [p.area for p in polygons]
            sorted_areas = sorted(areas, reverse=True)

            if len(sorted_areas) == 1:
                polygon = polygons[0]
            else:
                largest_area = sorted_areas[0]
                second_largest = sorted_areas[1]
                if largest_area > second_largest * multipolygon_rate:
                    polygon = polygons[areas.index(largest_area)]
                else:
                    print(
                        f"Skipping {name}: cannot determine main Polygon in MultiPolygon"
                    )
                    continue
        else:
            print(f"Skipping {name}: geometry type {type(geometry)} not supported")
            continue

        # Get bounds and create mask
        min_x, min_y, max_x, max_y = map(int, polygon.bounds)
        mask = PolygonProcessor.polygon_to_mask(polygon, im_shape)
        cropped_mask = mask[min_y:max_y, min_x:max_x]

        # Crop and mask each channel
        masked_cropped_im_dict = OrderedDict()
        for channel_name, im in im_dict.items():
            cropped_im = im[min_y:max_y, min_x:max_x]
            masked_cropped_im = np.where(cropped_mask, cropped_im, fill)
            masked_cropped_im_dict[channel_name] = masked_cropped_im
            pb.update(1)

        yield name, masked_cropped_im_dict


################################################################################
# Test
################################################################################


def main():
    from pyqupath.tiff import TiffZarrReader

    # Test plotting classification
    print("Test plotting classification")
    geojson_f = Path(__file__).parent.parent / "data/geojson/test.geojson"
    output_f = Path(__file__).parent.parent / "data/geojson/test_updated_1.geojson"

    name_dict = {"1": "Tumor", "2": "Stroma", "3": "Immune cells"}
    geojson_processor = GeojsonProcessor.from_path(geojson_f)
    geojson_processor.update_classification(name_dict)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ax = axs[0]
    geojson_processor.plot_classification(plot_raw=True, legend=False, ax=ax)
    ax.set_title("Raw")
    ax = axs[1]
    geojson_processor.plot_classification(ax=ax)
    ax.set_title("Updated")
    plt.tight_layout()
    plt.show()

    geojson_processor.output_geojson(output_f)

    # Test cropping array
    print("Test cropping array")
    tiff_f = Path(__file__).parent.parent / "data/ometiff/test_3d_pyramid.ome.tiff"
    geojson_f = Path(__file__).parent.parent / "data/geojson/test.geojson"

    tiff_reader = TiffZarrReader.from_ometiff(tiff_f)
    img = tiff_reader.zimg
    print(img.shape)

    for name, cropped_img in crop_array_by_geojson_batch(img, geojson_f):
        n_plot = min(6, len(cropped_img))
        fig, axs = plt.subplots(
            int(np.floor(np.sqrt(n_plot))),
            int(np.ceil(np.sqrt(n_plot))),
            figsize=(10, 10),
        )
        axs = axs.flatten()
        for ax in axs:
            ax.axis("off")
        for i in range(n_plot):
            ax = axs[i]
            ax.imshow(cropped_img[i], cmap="gray")
            ax.set_title(tiff_reader.channel_names[i])
        plt.tight_layout()
        plt.suptitle(f"Test cropping array: {name}")
        plt.show()

    # Test cropping dict
    print("Test cropping dict")
    tiff_f = Path(__file__).parent.parent / "data/ometiff/test_3d_pyramid.ome.tiff"
    geojson_f = Path(__file__).parent.parent / "data/geojson/test.geojson"

    tiff_reader = TiffZarrReader.from_ometiff(tiff_f)
    im_dict = tiff_reader.zimg_dict
    channel_names = tiff_reader.channel_names

    for name, cropped_im_dict in crop_dict_by_geojson_batch(im_dict, geojson_f):
        n_plot = min(6, len(cropped_im_dict))
        fig, axs = plt.subplots(
            int(np.floor(np.sqrt(n_plot))),
            int(np.ceil(np.sqrt(n_plot))),
            figsize=(10, 10),
        )
        axs = axs.flatten()
        for ax in axs:
            ax.axis("off")
        for i in range(n_plot):
            ax = axs[i]
            ax.imshow(cropped_im_dict[channel_names[i]], cmap="gray")
            ax.set_title(channel_names[i])
        plt.tight_layout()
        plt.suptitle(f"Test cropping dict: {name}")
        plt.show()


if __name__ == "__main__":
    main()

# %%
