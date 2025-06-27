# Co-developers:
# - Huaying Qiu: https://github.com/huayingq1996
# - Wenrui Wu: https://github.com/wuwenrui555

# %%
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_hex
from matplotlib.patches import Circle
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


# %%
def create_circular_kernel(radius):
    """
    Create a circular morphological kernel for image processing operations.

    This function generates a binary circular kernel that can be used for morphological
    operations like opening, closing, erosion, and dilation. The kernel is useful for
    operations that need to preserve circular features in images.

    Parameters
    ----------
    radius : int
        Radius of the circular kernel in pixels. Must be a positive integer.
        The actual kernel size will be (2*radius+1, 2*radius+1).

    Returns
    -------
    np.ndarray
        Binary circular kernel of shape (2*radius+1, 2*radius+1) with dtype uint8.
        Values are 1 inside the circle and 0 outside.

    Notes
    -----
    The kernel uses the standard circular equation: x² + y² ≤ r²
    to determine which pixels are inside the circle.
    """
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x * x + y * y <= radius * radius
    kernel = np.zeros((diameter, diameter), dtype=np.uint8)
    kernel[mask] = 1
    return kernel


def find_minimum_enclosing_circle(points):
    """
    Find the minimum enclosing circle for a set of 2D points using numerical optimization.

    This function implements an optimization-based approach to find the smallest circle
    that contains all given points. It uses the Nelder-Mead simplex algorithm to minimize
    the maximum violation (points outside the circle).

    Parameters
    ----------
    points : np.ndarray
        Array of 2D points with shape (n, 2), where n is the number of points.
        Each row represents a point with [x, y] coordinates.

    Returns
    -------
    tuple
        A tuple containing:
        - center : tuple of float
            (center_x, center_y) coordinates of the circle center
        - radius : float
            Radius of the minimum enclosing circle

    Notes
    -----
    The optimization minimizes the squared maximum violation to find the circle.
    Initial guess uses the bounding box center and maximum distance from center.
    Uses scipy.optimize.minimize with Nelder-Mead method for robustness.
    """

    def circle_error(params):
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        return np.max(distances - r) ** 2

    # Initial guess: center of all points and maximum distance
    center_x = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2
    center_y = (np.min(points[:, 1]) + np.max(points[:, 1])) / 2
    max_dist = np.max(
        np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)
    )

    initial_guess = [center_x, center_y, max_dist]

    # Optimize to find minimum enclosing circle
    result = minimize(circle_error, initial_guess, method="Nelder-Mead")

    return (result.x[0], result.x[1]), result.x[2]


def contours_to_gdf(contours):
    """
    Convert OpenCV contours to a GeoDataFrame with geometric properties.

    This function processes a list of OpenCV contours and converts them into a
    GeoDataFrame with additional geometric properties including area, center points,
    and minimum enclosing circles. Only contours with at least 3 points are processed.

    Parameters
    ----------
    contours : list
        List of OpenCV contours. Each contour should be a numpy array with shape
        (n_points, 1, 2) as returned by cv2.findContours().

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the following columns:
        - geometry : Shapely Polygon objects
        - center : Shapely Point objects representing polygon centroids
        - area : float, area of each polygon
        - center_x : float, x-coordinate of the center
        - center_y : float, y-coordinate of the center
        - radius : float, radius of minimum enclosing circle

        The GeoDataFrame is sorted by center coordinates (center_x, center_y).

    Notes
    -----
    - Contours with fewer than 3 points are skipped as they cannot form valid polygons
    - The minimum enclosing circle is calculated for each contour's boundary points
    - Results are automatically sorted by spatial coordinates for consistent ordering
    """
    data = []

    for i, contour in enumerate(contours):
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Create polygon geometry if contour has at least 3 points
        if len(contour) >= 3:
            # Convert contour to shapely polygon
            contour_points = contour.reshape(-1, 2)
            polygon = Polygon(contour_points)
            (center_x, center_y), radius = find_minimum_enclosing_circle(contour_points)
            data.append(
                {
                    "geometry": polygon,
                    "center": Point(center_x, center_y),
                    "area": area,
                    "center_x": center_x,
                    "center_y": center_y,
                    "radius": radius,
                }
            )

    # Create GeoDataFrame sorted by center coordinates
    gdf = (
        gpd.GeoDataFrame(data)
        .sort_values(["center_x", "center_y"])
        .reset_index(drop=True)
    )
    return gdf


def calculate_optimal_grid_size(gdf):
    """
    Calculate optimal grid size based on minimum distance between circle centers.

    This function analyzes the spatial distribution of circle centers to determine
    an appropriate grid size for spatial sorting. The grid size is set to 80% of
    the minimum distance between any two centers to ensure good spatial separation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing center_x and center_y columns with coordinate
        information for each geometric object.

    Returns
    -------
    float
        Optimal grid size for spatial sorting, calculated as 80% of the minimum
        distance between any two center points.

    Notes
    -----
    - Uses scipy.spatial.distance.pdist to calculate all pairwise distances
    - The 80% factor prevents over-segmentation while maintaining spatial structure
    - If all points are identical, the function will return 0
    - For single points, the function may raise an error from pdist
    """

    points = gdf[["center_x", "center_y"]].values
    distances = pdist(points)

    # Use 80% of minimum distance as grid size
    min_distance = np.min(distances)
    optimal_grid_size = min_distance * 0.8

    return optimal_grid_size


def create_grid_numbering(gdf, grid_size=100):
    """
    Create grid-based numbering system for spatial sorting of polygons.

    This function implements a spatial sorting algorithm that divides the coordinate
    space into a regular grid and sorts objects based on their grid position. Objects
    are sorted first by row (y-coordinate), then by column (x-coordinate), creating
    a consistent left-to-right, top-to-bottom ordering.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing center_x and center_y columns with coordinate
        information for spatial sorting.
    grid_size : float, default 100
        Size of grid cells for spatial sorting. Smaller values create finer
        spatial resolution but may be less robust to noise.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame sorted by grid coordinates with temporary grid columns removed.
        The index is reset to reflect the new spatial ordering.

    Notes
    -----
    - Grid coordinates are calculated using integer division (floor division)
    - Sorting is done by grid_y first (rows), then grid_x (columns)
    - Temporary grid columns are automatically removed from the output
    - This method is particularly useful for TMA (tissue microarray) core ordering
    """
    # Calculate grid coordinates
    gdf["grid_x"] = (gdf["center_x"] // grid_size).astype(int)
    gdf["grid_y"] = (gdf["center_y"] // grid_size).astype(int)

    # Sort by grid: first by row (y), then by column (x)
    gdf_sorted = gdf.sort_values(["grid_y", "grid_x"]).reset_index(drop=True)

    # Remove temporary columns
    gdf_sorted = gdf_sorted.drop(["grid_x", "grid_y"], axis=1)

    return gdf_sorted


def find_circle_overlaps(idx, center_x, center_y, radius, radius_expand=0):
    """
    Find overlapping circles and group them for merging.

    This function identifies circles that overlap based on their center coordinates
    and radii. Circles are considered overlapping if the distance between their
    centers is less than the sum of their radii. Overlapping circles are grouped
    together for potential merging operations.

    Parameters
    ----------
    idx : array-like
        Indices or identifiers of circles to analyze.
    center_x : array-like
        X coordinates of circle centers. Must have same length as idx.
    center_y : array-like
        Y coordinates of circle centers. Must have same length as idx.
    radius : array-like
        Radii of circles. Must have same length as idx.
    radius_expand : float, default 0
        Additional radius expansion for overlap detection. Positive values
        make overlap detection more sensitive (finds more overlaps).

    Returns
    -------
    list
        List of tuples, where each tuple contains indices of circles that
        overlap with each other. Single circles (no overlaps) are not included.

    Notes
    -----
    - Uses Euclidean distance for center-to-center calculations
    - Each circle can only belong to one merge group (greedy grouping)
    - The algorithm processes circles in the order they appear in idx
    - radius_expand can be used to create buffer zones around circles
    """
    df = pd.DataFrame(
        {"idx": idx, "center_x": center_x, "center_y": center_y, "radius": radius}
    ).set_index("idx")

    if len(df) == 0:
        return []

    merge_list = []
    used = set()
    indices = list(df.index)

    for idx in indices:
        if idx in used:
            continue

        # Start a new merge group
        merge_group = [idx]
        used.add(idx)

        # Get current circle information
        current_row = df.loc[idx]

        # Find all circles that intersect with current circle
        for other_idx in indices:
            if other_idx in used:
                continue

            other_row = df.loc[other_idx]

            # Calculate distance between circle centers
            distance = np.sqrt(
                (current_row["center_x"] - other_row["center_x"]) ** 2
                + (current_row["center_y"] - other_row["center_y"]) ** 2
            )

            # Check if circles overlap: distance < sum of radii
            if distance < (current_row["radius"] + other_row["radius"] + radius_expand):
                merge_group.append(other_idx)
                used.add(other_idx)

        # Add merge group to results if it contains multiple circles
        if len(merge_group) > 1:
            merge_list.append(tuple(merge_group))

    return merge_list


def merge_contours_and_find_circles(gdf, merge_list, radius_expand=0, grid_size=None):
    """
    Merge overlapping polygons and find minimum enclosing circles for all groups.

    This function performs the main merging operation for overlapping tissue cores.
    It combines polygon geometries that were identified as overlapping, calculates
    new minimum enclosing circles for merged groups, and maintains individual
    polygons that don't overlap with others. The results are spatially sorted
    and given sequential identifiers.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Original GeoDataFrame with polygon geometries and associated properties.
        Must contain 'geometry' and 'area' columns.
    merge_list : list
        List of tuples containing indices of polygons to merge together.
        Each tuple represents one merge group.
    radius_expand : float, default 0
        Additional radius expansion applied to all final circles. Useful for
        adding buffer zones around detected cores.
    grid_size : float, optional
        Grid size for spatial sorting. If None, calculated automatically using
        calculate_optimal_grid_size().

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with merged polygons and their properties:
        - merge_group : list of original indices in each group
        - geometry : merged polygon geometries (or original for single polygons)
        - center_x, center_y : coordinates of minimum enclosing circle centers
        - radius : radius of minimum enclosing circles (with expansion applied)
        - total_area : sum of areas in each merge group
        - is_merged : boolean indicating whether polygon was created by merging

        Index is set to sequential identifiers like 'c000', 'c001', etc.

    Notes
    -----
    - Merged geometries use shapely.ops.unary_union for robust polygon combination
    - Minimum enclosing circles are recalculated for all boundary points in merge groups
    - Spatial sorting ensures consistent ordering across different runs
    - Sequential naming uses zero-padded numbers based on total count
    """
    merged_results = []

    # Collect all indices that are in merge_list
    merged_indices = set()
    for merge_group in merge_list:
        merged_indices.update(merge_group)

    # Process polygons in merge_list (merge them)
    for merge_group in merge_list:
        polygons_to_merge = gdf[gdf.index.isin(merge_group)]
        if len(polygons_to_merge) == 0:
            continue

        # Merge all polygon geometries
        merged_geometry = unary_union(polygons_to_merge.geometry.tolist())

        # Find minimum enclosing circle for all polygons
        all_points = []
        for idx in merge_group:
            polygon = gdf.loc[idx, "geometry"]
            coords = list(polygon.exterior.coords)
            all_points.extend(coords)
        all_points = np.array(all_points)

        (center_x, center_y), radius = find_minimum_enclosing_circle(all_points)

        merged_results.append(
            {
                "merge_group": merge_group,
                "geometry": merged_geometry,
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius,
                "total_area": polygons_to_merge["area"].sum(),
                "is_merged": True,
            }
        )

    # Process individual polygons not in merge_list
    for idx, row in gdf.iterrows():
        if idx not in merged_indices:
            # Find minimum circle for single polygon
            polygon = row["geometry"]
            coords = list(polygon.exterior.coords)
            all_points = np.array(coords)

            (center_x, center_y), radius = find_minimum_enclosing_circle(all_points)

            merged_results.append(
                {
                    "merge_group": [idx],
                    "geometry": polygon,
                    "center_x": center_x,
                    "center_y": center_y,
                    "radius": radius,
                    "total_area": row["area"],
                    "is_merged": False,
                }
            )

    # Create final GeoDataFrame
    gdf = (
        gpd.GeoDataFrame(merged_results)
        .sort_values(["center_x", "center_y"])
        .reset_index(drop=True)
    )
    gdf["radius"] = gdf["radius"] + radius_expand

    # Apply grid-based sorting
    if grid_size is None:
        optimal_grid_size = calculate_optimal_grid_size(gdf)
        gdf = create_grid_numbering(gdf, grid_size=optimal_grid_size)
    else:
        gdf = create_grid_numbering(gdf, grid_size=grid_size)

    # Create sequential naming
    total_count = len(gdf)
    num_digits = len(str(total_count))
    new_index = [f"c{i:0{num_digits}d}" for i in range(total_count)]
    gdf.index = new_index

    return gdf


def plot_contours(
    gdf,
    cmap="tab20",
    plot_center=False,
    plot_id=True,
    plot_circle=True,
    merge_list=None,
    text_size=12,
    text_color="black",
    figsize=(10, 10),
):
    """
    Plot contours with optional centers, IDs, and circles for visualization.

    This function creates a comprehensive visualization of detected tissue cores
    with various display options. It's particularly useful for quality control
    and parameter tuning during the dearraying process.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing polygon geometries to plot. Must have 'geometry',
        'center_x', 'center_y', and 'radius' columns.
    cmap : str, default "tab20"
        Matplotlib colormap name for polygon colors. "tab20" provides 20 distinct
        colors that cycle for larger datasets.
    plot_center : bool, default False
        Whether to plot center points as red dots on the visualization.
    plot_id : bool, default True
        Whether to plot polygon IDs as text labels at each center.
    plot_circle : bool, default True
        Whether to plot minimum enclosing circles around each polygon.
    merge_list : list, optional
        List of polygon groups that will be merged. If provided, circles for
        these groups are highlighted in red, others in green.
    text_size : int, default 12
        Font size for ID labels when plot_id=True.
    text_color : str, default "black"
        Color for ID label text.
    figsize : tuple, default (10, 10)
        Figure size as (width, height) in inches.

    Returns
    -------
    tuple
        (fig, ax) - matplotlib figure and axes objects for further customization.

    Notes
    -----
    - Colors cycle through the colormap when there are more polygons than colors
    - Y-axis is inverted to match image coordinate systems (origin at top-left)
    - Equal aspect ratio is maintained for accurate shape representation
    - Circles are drawn with no fill, only colored edges for clear polygon visibility
    """
    # Get colormap and assign colors to polygons
    n_polygons = len(gdf)
    colors = []
    for i in range(n_polygons):
        color_idx = i % 20  # tab20 has 20 colors, cycle through them
        color = colormaps[cmap](color_idx)
        colors.append(to_hex(color))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot polygons with assigned colors
    gdf.plot(
        ax=ax,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Optional: add center points
    if plot_center:
        gdf.set_geometry("center").plot(ax=ax, color="red", markersize=10, alpha=0.8)

    # Optional: add ID labels
    if plot_id:
        for idx, row in gdf.iterrows():
            ax.annotate(
                str(idx),
                (row.center_x, row.center_y),
                ha="center",
                va="center",
                fontsize=text_size,
                color=text_color,
                weight="bold",
            )

    # Optional: add enclosing circles
    if plot_circle:
        if merge_list is not None:
            flattened_merge_list = [
                idx for group in merge_list if len(group) > 1 for idx in group
            ]
        else:
            flattened_merge_list = []

        circles = []
        colors_circle = []

        for idx, row in gdf.iterrows():
            circles.append(Circle((row["center_x"], row["center_y"]), row["radius"]))

            # Highlight circles that will be merged
            if idx in flattened_merge_list:
                colors_circle.append("red")
            else:
                colors_circle.append("green")

        circle_collection = PatchCollection(
            circles,
            facecolors="none",
            edgecolors=colors_circle,
            linewidths=1,
            linestyles="-",
        )
        ax.add_collection(circle_collection)

    ax.invert_yaxis()
    plt.title(f"Filtered Contours (n={len(gdf)})")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    return fig, ax


def restore_to_original_scale(gdf, downsample_step=16):
    """
    Restore downsampled coordinates and radii to original image scale.

    This function converts processing results from downsampled coordinates back
    to the original full-resolution image scale. It generates new circular
    geometries at the original scale to replace the processed polygon geometries.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame based on downsampled image processing. Must contain
        'center_x', 'center_y', and 'radius' columns.
    downsample_step : int, default 16
        Downsampling factor used during processing. All coordinates and radii
        are multiplied by this factor to restore original scale.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with coordinates and geometries restored to original image scale:
        - geometry : Circular polygons at original scale
        - name : Original index names from input GeoDataFrame
        - center_x, center_y : Center coordinates at original scale
        - radius : Radii at original scale

    Notes
    -----
    - Generates circular polygons with 360 degree resolution (1-degree steps)
    - Original polygon shapes are replaced with perfect circles
    - All spatial measurements are multiplied by the downsample_step factor
    - Useful for applying results to original high-resolution images
    """
    # Scale coordinates and radii by downsampling factor
    names = gdf.index
    center_x_list = gdf["center_x"] * downsample_step
    center_y_list = gdf["center_y"] * downsample_step
    radius_list = gdf["radius"] * downsample_step

    # Generate circular geometries at original scale
    geometry_list = []
    for center_x, center_y, radius in zip(center_x_list, center_y_list, radius_list):
        # Generate circle points
        angles = np.arange(0, 360, 1)
        angles_rad = np.radians(angles)

        x_points = center_x + radius * np.cos(angles_rad)
        y_points = center_y + radius * np.sin(angles_rad)

        circle_points = list(zip(x_points, y_points))
        circle_polygon = Polygon(circle_points)
        geometry_list.append(circle_polygon)

    return gpd.GeoDataFrame(
        {
            "geometry": geometry_list,
            "name": names,
            "center_x": center_x_list,
            "center_y": center_y_list,
            "radius": radius_list,
        }
    )


# Main processing pipeline
def tma_dearrayer(
    img_dapi,
    downsample_step=16,
    kernel_radius=5,
    area_threshold=1000,
    radius_expand=5,
    merge_list=None,
    grid_size=None,
):
    """
    Complete pipeline for TMA (Tissue Microarray) core detection and dearraying.

    This is the main function that orchestrates the entire dearraying process.
    It takes a DAPI-stained tissue microarray image and automatically detects
    individual tissue cores, handles overlapping cores through merging, and
    returns organized core information with spatial sorting.

    The pipeline includes: image downsampling for efficiency, Otsu thresholding
    for tissue detection, morphological operations for noise reduction, contour
    detection, overlap analysis, merging of overlapping cores, and spatial sorting.

    Parameters
    ----------
    img_dapi : np.ndarray
        Input DAPI-stained image as a 2D numpy array. Should be grayscale with
        tissue areas appearing brighter than background.
    downsample_step : int, default 16
        Downsampling factor for processing efficiency. Larger values process
        faster but may miss fine details. Recommended range: 8-32.
    kernel_radius : int, default 5
        Radius for morphological closing operations. Larger values connect
        nearby tissue fragments but may merge distinct cores. Range: 3-10.
    area_threshold : float, default 1000
        Minimum area threshold for filtering small contours (in downsampled pixels).
        Helps remove noise and artifacts. Adjust based on expected core sizes.
    radius_expand : float, default 5
        Additional radius expansion for final circles (in downsampled pixels).
        Creates buffer zones around cores. Useful for ensuring complete coverage.
    merge_list : list, optional
        Pre-defined list of core indices to merge. If None, overlaps are
        detected automatically. Format: [(idx1, idx2), (idx3, idx4, idx5), ...]
    grid_size : float, optional
        Grid size for spatial sorting (in downsampled pixels). If None,
        calculated automatically based on core spacing.

    Returns
    -------
    tuple
        A tuple containing five elements:
        - gdf_filtered : gpd.GeoDataFrame
            Initial filtered contours before merging
        - merge_list : list
            List of detected or provided merge groups
        - gdf_merge : gpd.GeoDataFrame
            Final processed cores after merging (downsampled scale)
        - overlap_list : list
            List of any remaining overlapping circles after merging
        - gdf_original : gpd.GeoDataFrame
            Final cores scaled back to original image resolution

    Notes
    -----
    - Processing is done on downsampled images for efficiency, then scaled back
    - Otsu thresholding automatically determines tissue/background threshold
    - Morphological closing helps connect fragmented tissue regions
    - Overlap detection uses circle-circle intersection analysis
    - Spatial sorting provides consistent core ordering for analysis
    - All coordinates in gdf_original are at the original image resolution
    - Function prints progress information about merge groups and overlaps
    """
    # Validate input
    if img_dapi.ndim != 2:
        raise ValueError("img_dapi must be a 2D numpy array")

    # Downsample image
    img = img_dapi[::downsample_step, ::downsample_step]

    # Apply Otsu thresholding
    _, otsu = cv2.threshold(img, 0, 2**16 - 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill gaps
    kernel = create_circular_kernel(kernel_radius)
    otsu_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    mask_8bit = (otsu_close > 0).astype(np.uint8)

    # Find contours
    contours, hierarchy = cv2.findContours(
        mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert to GeoDataFrame and filter by area
    gdf = contours_to_gdf(contours)
    gdf_filtered = gdf[gdf.area >= area_threshold]

    # Find overlapping circles
    if merge_list is None:
        merge_list = find_circle_overlaps(
            gdf_filtered.index,
            gdf_filtered["center_x"],
            gdf_filtered["center_y"],
            gdf_filtered["radius"],
        )
    print(f"Found {len(merge_list)} groups to merge:")
    if len(merge_list) > 0:
        for i, group in enumerate(merge_list):
            print(f"    Group {i + 1}: {group}")

    # Merge contours and find final circles
    gdf_merge = merge_contours_and_find_circles(
        gdf_filtered, merge_list, radius_expand=radius_expand, grid_size=grid_size
    )

    # Check if any circles were overlapping
    overlap_list = find_circle_overlaps(
        gdf_merge.index,
        gdf_merge.center_x,
        gdf_merge.center_y,
        gdf_merge.radius,
    )
    print(f"Found {len(overlap_list)} overlapping circles:")
    if len(overlap_list) > 0:
        for i, group in enumerate(overlap_list):
            print(f"    Group {i + 1}: {group}")

    # Restore to original scale
    gdf_original = restore_to_original_scale(gdf_merge, downsample_step)

    return gdf_filtered, merge_list, gdf_merge, overlap_list, gdf_original
