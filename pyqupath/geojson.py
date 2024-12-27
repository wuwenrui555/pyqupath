import json

import geopandas as gpd


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
