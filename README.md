# pyqupath

## `geojson`

- `load_geojson_to_gdf()`: Load a GeoJSON file or string as GeoPandas GeoDataFrame.
- `polygon_to_mask()`: Generate a binary mask from a polygon.
- `mask_to_geojson()`: Convert a labeled mask into a GeoJSON file.
- `mask_to_geojson_joblib()`: Convert a labeled mask into a GeoJSON file using parallel processing.
- `crop_dict_by_geojson()`: Crop regions from images in a dictionary using polygons from a GeoJSON file.
- `update_geojson_classification()`: Update classification names and colors in a GeoJSON file.

## `ometiff`

### pyramidal OME-TIFF writer

- `export_ometiff_pyramid()`: Generate a pyramidal OME-TIFF file from multiple input TIFF files.
- `export_ometiff_pyramid_from_dict()`: Generate a pyramidal OME-TIFF file from a dictionary of images.
- `export_ometiff_pyramid_from_qptiff()`: Generate a pyramidal OME-TIFF file from a QPTIFF file.

### metadata extraction

- `extract_channels_from_ometiff()`: Extract channel names from an OME-TIFF file.
- `extract_channels_from_qptiff()`: Extract channel names from a QPTIFF file.

### tiff reader

- `tiff_highest_resolution_generator()`: Generator to read the highest resolution level of a multi-page TIFF file.
- `load_tiff_to_dict()`: Load a multi-channel TIFF file into a dictionary of channel images.
