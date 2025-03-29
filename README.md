# pyqupath

A Python package for processing and analyzing microscopy images with QuPath integration.

## Features

### `geojson`

#### IO and Processing

- `GeojsonProcessor` class: A class for reading and manipulating GeoJSON files
  - Factory methods:
    - `from_path()`: Create processor from GeoJSON file
    - `from_text()`: Create processor from GeoJSON text
  - Properties:
    - `gdf`: The GeoDataFrame containing the GeoJSON data
  - Methods:
    - `set_index()`: Set index with duplicate handling
    - `plot_classification()`: Plot classifications with legend
    - `update_classification()`: Update classification names and colors
    - `output_geojson()`: Output the GeoDataFrame as a GeoJSON file

```python
from pyqupath.geojson import GeojsonProcessor
from pathlib import Path

# Initialize processor
geojson_f = Path("/path/to/input.geojson")
output_f = Path("/path/to/output.geojson")
geojson_processor = GeojsonProcessor.from_path(geojson_f)

# Plot raw classification
geojson_processor.plot_classification()

# Update classification
name_dict = {"1": "Tumor", "2": "Stroma", "3": "Immune cells"}
geojson_processor.update_classification(name_dict)
geojson_processor.plot_classification()

# Output updated GeoJSON
geojson_processor.output_geojson(output_f)
```

#### Polygon Processing

- `PolygonProcessor` class: A class for processing polygon geometries
  - Methods:
    - `polygon_to_mask()`: Generate a binary mask from a polygon
    - `crop_array_by_polygon()`: Crop an image using a polygon and apply a mask
- `crop_array_by_polygon_batch()`: Crop an image using a list of polygons
- `crop_dict_by_polygon_batch()`: Crop a dictionary of images by a list of polygons

```python
from pyqupath.geojson import crop_array_by_geojson_batch, crop_dict_by_geojson_batch
from pyqupath.tiff import TiffZarrReader
from pathlib import Path

# Initialize reader
tiff_f = Path("/path/to/input.ome.tiff")
geojson_f = Path("/path/to/input.geojson")
tiff_reader = TiffZarrReader.from_ometiff(tiff_f)

# Cropping array
img = tiff_reader.zimg
for name, cropped_img in crop_array_by_geojson_batch(img, geojson_f):
    print(f"{name}: {cropped_img.shape}")

# Cropping dict
img_dict = tiff_reader.zimg_dict
for name, cropped_im_dict in crop_dict_by_geojson_batch(img_dict, geojson_f):
    print(f"{name}: {len(cropped_im_dict)}")
```

#### Mask Conversion

- `binary_mask_to_polygon()`: Convert a binary mask to a Shapely Polygon
- `mask_to_polygon_batch()`: Convert a batch of labels from a mask into Polygons
- `mask_to_polygons()`: Convert a segmentation mask into a list of Polygons
- `mask_to_geojson()`: Convert a labeled mask into a GeoJSON file with optional simplification
- `mask_to_geojson_joblib()`: Convert a labeled mask into a GeoJSON file using parallel processing

### `tiff`

#### TIFF Reader

Lazy loading of images using zarr, support for both OME-TIFF and QPTIFF formats, channel-based indexing, and region-based loading

- `TiffZarrReader` class: A class for efficient reading of TIFF files with these features:
  - Factory methods:
    - `from_ometiff()`: Create reader from OME-TIFF file
    - `from_qptiff()`: Create reader from QPTIFF file
  - Properties:
    - `zimg`: The image as a zarr array
    - `zimg_dict`: The image as a dictionary of zarr arrays
    - `channel_names`: The names of the channels

#### Pyramidal OME-TIFF Writer

Supports multi-resolution pyramid generation, parallel processing, progress tracking, and flexible input formats

- `PyramidWriter` class: A class for generating pyramidal OME-TIFF files with these features:
  - Factory methods:
    - `from_array()`: Create writer from numpy/zarr array
    - `from_dict()`: Create writer from dictionary of arrays
    - `from_fs()`: Create writer from file system paths
  - Properties:
    - `zimg`: The image as a zarr array
    - `zimg_dict`: The image as a dictionary of zarr arrays
    - `channel_names`: The names of the channels
  - Methods:
    - `export_ometiff_pyramid()`: Generate pyramidal OME-TIFF with customizable parameters

```python
from pyqupath.tiff import TiffZarrReader, PyramidWriter
from pathlib import Path

# Read and write pyramidal OME-TIFF
ometiff_f = Path("/path/to/input.ome.tiff")
output_f = Path("/path/to/output.ome.tiff")

tiff_reader = TiffZarrReader.from_ometiff(ometiff_f)
tiff_writer = PyramidWriter.from_array(tiff_reader.zimg)
tiff_writer.export_ometiff_pyramid(output_f=output_f)
```

### `buffer`

#### Line Processing

- `add_buffers()`: Add buffer zones around line geometries
- `merge_lines()`: Merge two lines if their buffers intersect
- `merge_buffers()`: Merge intersecting buffers of lines and polygons

#### Visualization

- `plot_geometries()`: Plot geometries with unique colors
- `plot_lines_polygons()`: Plot polygons and lines with optional buffers

## Deprecated

The following modules and functions are deprecated and will be removed in future versions. Please use the recommended alternatives.

### `geojson`

The following functions are deprecated. Please use the `GeojsonProcessor` class instead:

- `load_geojson_to_gdf()`: Use `GeojsonProcessor.from_path()` or `GeojsonProcessor.from_text()`
- `update_geojson_classification()`: Use `GeojsonProcessor.update_classification()`
- `crop_dict_by_geojson()`: Use `crop_dict_by_polygon_batch()`

### `ometiff` (use `tiff` instead)

The following functions are deprecated. Please use the `TiffZarrReader` and `PyramidWriter` classes instead:

#### Pyramidal OME-TIFF Writer

- `export_ometiff_pyramid()`: Use `PyramidWriter.from_array()`
- `export_ometiff_pyramid_from_dict()`: Use `PyramidWriter.from_dict()`
- `export_ometiff_pyramid_from_qptiff()`: Use `PyramidWriter.from_fs()`

#### Metadata Extraction

- `extract_channels_from_ometiff()`: Use `TiffZarrReader.from_ometiff()`
- `extract_channels_from_qptiff()`: Use `TiffZarrReader.from_qptiff()`

#### TIFF Reader

- `tiff_highest_resolution_generator()`: Use `TiffZarrReader` with region-based loading
- `load_tiff_to_dict()`: Use `TiffZarrReader` with channel-based indexing
