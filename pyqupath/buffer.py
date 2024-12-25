import itertools

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

################################################################################
# Merge open polylines
################################################################################


def add_buffers(
    gdf: gpd.GeoDataFrame,
    buffer_distance: float,
) -> gpd.GeoDataFrame:
    """
    Adds buffer zones around the beginning and end points of line geometries

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing line geometries in the `geometry` column.
    buffer_distance : float
        The radius of the buffer to create around the beginning and end points of each line.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with two additional columns:
        - `beg_buffer`: GeoSeries containing the buffer geometries around the beginning points.
        - `end_buffer`: GeoSeries containing the buffer geometries around the end points.
    """
    beg_buffers = []
    end_buffers = []
    for line in gdf.geometry:
        beg_point = Point(line.coords[0])
        end_point = Point(line.coords[-1])
        beg_buffers.append(beg_point.buffer(buffer_distance))
        end_buffers.append(end_point.buffer(buffer_distance))

    gdf["beg_buffer"] = gpd.GeoSeries(beg_buffers)
    gdf["end_buffer"] = gpd.GeoSeries(end_buffers)
    return gdf


def plot_geometries(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
) -> plt.Axes:
    """
    Plot geometries (polygons and lines) from a GeoDataFrame with unique colors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the geometries will be plotted.
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the geometries to plot. Each geometry will be plotted with a unique color.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plotted geometries.
    """
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = itertools.cycle(default_colors)

    for i in gdf.index:
        color = next(color_cycle)
        gdf.geometry.loc[[i]].plot(ax=ax, color=color)
    return ax


def plot_lines_polygons(
    gdf_polygons: gpd.GeoDataFrame,
    gdf_lines: gpd.GeoDataFrame,
    buffer_distance: float = 50,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    figsize: tuple[float, float] = (10, 10),
) -> plt.Figure:
    """
    Plot polygons and lines with optional buffers

    Parameters
    ----------
    gdf_polygons : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries to plot.
    gdf_lines : geopandas.GeoDataFrame
        GeoDataFrame containing line geometries to plot.
    buffer_distance : float, optional
        The distance to use when generating buffers around the lines (default is 50).
    xlim : tuple of float, optional
        Limits for the x-axis in the format (xmin, xmax). If None, no limit is set.
    ylim : tuple of float, optional
        Limits for the y-axis in the format (ymin, ymax). If None, no limit is set.
    figsize : tuple of float, optional
        Size of the figure in inches as (width, height). Default is (10, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plotted geometries.

    Notes
    -----
    - Polygons are plotted in red, while lines are plotted with unique colors.
    - Buffers around lines are plotted with semi-transparent red (start buffers) and blue (end buffers).
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes = plot_geometries(axes, gdf_polygons)
    axes = plot_geometries(axes, gdf_lines)
    gdf_lines_buffers = add_buffers(gdf_lines.copy(), buffer_distance)
    gdf_lines_buffers.beg_buffer.plot(ax=axes, color="red", alpha=0.2)
    gdf_lines_buffers.end_buffer.plot(ax=axes, color="blue", alpha=0.2)

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    plt.close(fig)
    return fig


def merge_lines(
    i_gdf_line: gpd.GeoSeries,
    j_gdf_line: gpd.GeoSeries,
) -> LineString:
    """
    Merge two GeoDataFrame lines if their buffers intersect.

    Parameters
    ----------
    i_gdf_line : GeoSeries
        The first line with buffer attributes `beg_buffer` and `end_buffer`.
    j_gdf_line : GeoSeries
        The second line with buffer attributes `beg_buffer` and `end_buffer`.

    Returns
    -------
    LineString or bool
        A merged LineString if the buffers intersect; otherwise, False.
    """
    if i_gdf_line.beg_buffer.intersects(j_gdf_line.beg_buffer):
        merged_coords = list(i_gdf_line.geometry.coords[::-1]) + list(
            j_gdf_line.geometry.coords
        )
        return LineString(merged_coords)
    elif i_gdf_line.beg_buffer.intersects(j_gdf_line.end_buffer):
        merged_coords = list(i_gdf_line.geometry.coords[::-1]) + list(
            j_gdf_line.geometry.coords[::-1]
        )
        return LineString(merged_coords)
    elif i_gdf_line.end_buffer.intersects(j_gdf_line.beg_buffer):
        merged_coords = list(i_gdf_line.geometry.coords) + list(
            j_gdf_line.geometry.coords
        )
        return LineString(merged_coords)
    elif i_gdf_line.end_buffer.intersects(j_gdf_line.end_buffer):
        merged_coords = list(i_gdf_line.geometry.coords) + list(
            j_gdf_line.geometry.coords[::-1]
        )
        return LineString(merged_coords)
    else:
        return False


def merge_buffers(
    gdf_polygons: gpd.GeoDataFrame,
    gdf_lines: gpd.GeoDataFrame,
    buffer_distance: float = 50,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Merge intersecting buffers of lines and polygons in GeoDataFrames.

    Parameters
    ----------
    gdf_polygons : GeoDataFrame
        GeoDataFrame containing polygon geometries to retain.
    gdf_lines : GeoDataFrame
        GeoDataFrame containing line geometries to be merged based on buffer intersections.
    buffer_distance : float, optional
        Buffer distance to create around line geometries (default is 50).

    Returns
    -------
    tuple of GeoDataFrame
        A tuple containing two GeoDataFrames:
        - Merged polygons (original and newly formed from intersecting lines).
        - Remaining or merged lines.
    """
    gdf_lines_0 = []
    gdf_polygons_0 = []
    gdf_lines = add_buffers(gdf_lines, buffer_distance=buffer_distance)

    # merge self polygons
    for i in gdf_lines.index:
        i_beg_buffer = gdf_lines.at[i, "beg_buffer"]
        i_end_buffer = gdf_lines.at[i, "end_buffer"]
        if i_beg_buffer.intersects(i_end_buffer):
            gdf_polygons_0.append(Polygon(gdf_lines.at[i, "geometry"]))
            gdf_lines.drop(i, inplace=True)
    gdf_lines = gdf_lines.reset_index(drop=True)

    # merge lines
    index_merged = []
    for i in gdf_lines.index:
        if i in index_merged:
            continue
        i_gdf_line = gdf_lines.loc[i]
        i_merged = False
        j_indecies = [
            index
            for index in gdf_lines.index
            if (index not in index_merged) and (index > i)
        ]
        for j in j_indecies:
            j_gdf_line = gdf_lines.loc[j]
            merged_line = merge_lines(i_gdf_line, j_gdf_line)
            if merged_line:
                gdf_lines_0.append(merged_line)
                i_merged = True
                index_merged.append(j)
                break
        if not i_merged:
            gdf_lines_0.append(i_gdf_line.geometry)

    gdf_lines_0 = gpd.GeoDataFrame(
        gdf_lines_0,
        columns=["geometry"],
        crs=gdf_lines.crs,
    )
    gdf_polygons_0 = gpd.GeoDataFrame(
        gdf_polygons_0,
        columns=["geometry"],
        crs=gdf_lines.crs,
    )

    return (
        pd.concat(
            [gdf_polygons, gdf_polygons_0],
            ignore_index=True,
        ).set_geometry("geometry"),
        gdf_lines_0.set_geometry("geometry"),
    )
