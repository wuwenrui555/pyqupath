# %%
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import pathlib
import re
import sys
import uuid
from pathlib import Path
from typing import Union
from xml.etree import ElementTree

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform
import tifffile
import zarr
from shapely.geometry import Polygon
from tqdm import tqdm

from pyqupath.geojson import load_geojson_to_gdf, polygon_to_mask

# from cchalign import constants
# TQDM_FORMAT = constants.TQDM_FORMAT
TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

################################################################################
# TIFF Reader
################################################################################


class TiffZarrReader:
    """
    A class for reading TIFF files with advanced features using zarr.

    This class provides functionality to:
    1. Read OME-TIFF and QPTIFF files with marker name indexing
    2. Load specific regions of images without loading the entire image
    3. Lazy loading of images (only load when accessed)
    """

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        filetype: str = None,
    ):
        """
        Initialize the TIFF reader.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the TIFF file
        filetype : str, optional
            Type of TIFF file ('ome.tiff' or 'qptiff'). If None, will be inferred
            from the file extension.
        """
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        # Determine file type
        if filetype is None:
            filetype = str(self.path).split(".", 1)[-1]
        if filetype not in ["ome.tiff", "qptiff"]:
            raise ValueError(f"{filetype} is not supported right now.")

        # Initialize zarr reader
        self.zimg = zarr.open(tifffile.imread(ometiff_f, level=0, aszarr=True))

        # Get channel names
        if self.zimg.ndim == 3:
            if filetype == "ome.tiff":
                _extract_channels = self.extract_channels_from_ometiff
            elif filetype == "qptiff":
                _extract_channels = self.extract_channels_from_qptiff
            try:
                self.channel_names = _extract_channels(self.path)
            except Exception:
                self.channel_names = [f"channel_{i}" for i in range(self.zimg.shape[0])]
        elif self.zimg.ndim == 2:
            self.channel_names = ["channel_0"]
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.zimg.ndim}")

        # zimg_dict is a dictionary of zarr arrays, indexed by channel name
        self.zimg_dict = {
            channel_name: zarr.open(
                tifffile.imread(ometiff_f, key=i, level=0, aszarr=True)
            )
            for i, channel_name in enumerate(self.channel_names)
        }

    def channel_index(self, channels: Union[str, list[str]]) -> Union[int, list[int]]:
        """
        Get the index of a channel or a list of channels.

        Parameters
        ----------
        channels : str or list[str]
            The channel name or list of channel names to get the index of.

        Returns
        -------
        int or list[int]
            The index of the channel or a list of indices.
        """
        if self.channel_names is None:
            raise ValueError("Channel names not found")

        if isinstance(channels, str):
            return self.channel_names.index(channels)
        elif isinstance(channels, list):
            return [self.channel_names.index(ch) for ch in channels]
        else:
            raise ValueError(f"Invalid channel type: {type(channels)}")

    @staticmethod
    def extract_channels_from_ometiff(path: Union[str, pathlib.Path]) -> list[str]:
        """
        Extract channel names from an OME-TIFF file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the OME-TIFF file

        Returns
        -------
        list of str
            A list of channel names.
        """
        with tifffile.TiffFile(path) as tif:
            ome_metadata = ElementTree.fromstring(tif.ome_metadata)
            ome_channels = ome_metadata.findall(".//{*}Channel")
            metadata = pd.DataFrame([channel.attrib for channel in ome_channels])
            if "Name" in metadata.columns:
                channels = metadata["Name"].tolist()
            else:
                channels = [f"Channel {i}" for i in range(len(metadata))]
        return channels

    @staticmethod
    def extract_channels_from_qptiff(path: Union[str, pathlib.Path]) -> list[str]:
        """
        Extract channel names from a QPTIFF file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the QPTIFF file

        Returns
        -------
        list of str
            A list of channel names.
        """
        with tifffile.TiffFile(path) as im:
            xml_string = im.series[0].pages[0].tags["ImageDescription"].value
            scan_profile = ElementTree.fromstring(xml_string).find(".//ScanProfile")
            if scan_profile is None:
                raise ValueError(
                    "ScanProfile element not found in the provided XML string."
                )

            scan_profile_data = json.loads(scan_profile.text)
            wells = scan_profile_data.get("experimentDescription").get("wells")
            qptiff_metadata = pd.concat(
                [
                    pd.DataFrame(well.get("items")).assign(
                        wellName=well.get("wellName")
                    )
                    for well in wells
                ],
                ignore_index=True,
            )
            is_marker = (qptiff_metadata["markerName"] != "--") & (
                qptiff_metadata["id"].apply(
                    lambda x: re.search(r"^0+(-0+)+$", x.strip()) is None
                )
            )
            channels = (
                qptiff_metadata.loc[is_marker]
                .drop_duplicates(["id", "markerName"])["markerName"]
                .tolist()
            )
        return channels


################################################################################
# TIFF Writer
################################################################################


def export_ometiff_pyramid(
    input_data: Union[list[Union[str, pathlib.Path]], dict[str, np.ndarray]],
    output_f: Union[str, pathlib.Path],
    channel_names: list[str] = None,
    pixel_size: float = None,
    tile_size: int = 256,
    is_mask: bool = False,
    num_threads: int = 0,
    overwrite: bool = True,
):
    """
    Assemble a pyramidal OME-TIFF file.

    Parameters
    ----------
    input_data : list of str or dict of str to np.ndarray
        A list of file paths to the input TIFF images or a dictionary where keys
        are channel names and values are 2D numpy arrays representing the images.
        All images must have the same dimensions and pixel type.
    output_f : str or pathlib.Path
        Path to the output OME-TIFF file.
    channel_names : list of str
        Names of the channels in the OME-TIFF file. Each name corresponds to a
        channel in the `input_data`. The length of this list must match the number
        of files in `input_data`. Default is None.
    pixel_size : float, optional
        Pixel size in microns. Will be recorded in OME-XML metadata.
    tile_size : int, optional
        Width of pyramid tiles in output file (must be a multiple of 16).
        Default is 256.
    is_mask : bool, optional
        Adjust processing for label mask or binary mask images (currently just
        switch to nearest-neighbor downsampling). Default if False.
    num_threads : int, optional
        Number of parallel threads to use for image downsampling. Default is
        number of available CPUs.
    overwrite : bool, optional
        If True, the function will overwrite the output file if it already exists.
        If False, the function will terminate to prevent overwriting. Default
        is True.
    """

    def _error(path, msg):
        """
        Print an error message and exit the program.
        """
        print(f"\nERROR: {path}: {msg}")
        sys.exit(1)

    def _image_validation(
        shape: tuple[int, int],
        dtype: np.dtype,
        target_shape: tuple[int, int],
        is_mask: bool = False,
        msg_tag: str = None,
    ):
        """
        Validate the shape and dtype of an image.
        """
        if dtype == np.uint32 or dtype == np.int32:
            if not is_mask:
                _error(
                    msg_tag,
                    "32-bit images are only supported in is_mask = True. "
                    "Please contact the authors if you need support for "
                    "intensity-based 32-bit images.",
                )
        elif dtype == np.uint8 or dtype == np.uint16:
            pass
        else:
            _error(
                msg_tag,
                f"Can't handle dtype '{dtype}' yet, please contact the authors.",
            )
        if shape != target_shape:
            _error(
                msg_tag,
                f"Expected shape {target_shape} to match first input image,"
                f" got {shape} instead.",
            )

    output_f = pathlib.Path(output_f)
    if output_f.exists():
        if overwrite:
            print(f"Overwriting existing file: {output_f}")
            output_f.unlink()
        else:
            _error(output_f, "Output file already exists, remove before continuing.")

    if num_threads == 0:
        if hasattr(os, "sched_getaffinity"):
            num_threads = len(os.sched_getaffinity(0))
        else:
            num_threads = multiprocessing.cpu_count()
        print(f"Using {num_threads} worker threads based on detected CPU count.")
        print()
    tifffile.TIFF.MAXWORKERS = num_threads
    tifffile.TIFF.MAXIOWORKERS = num_threads * 5

    if isinstance(input_data, dict):
        if channel_names is None:
            channel_names = list(input_data.keys())
        in_imgs = [input_data[channel_name] for channel_name in channel_names]

        target_shape = in_imgs[0].shape
        for channel_name, img_in in zip(channel_names, in_imgs):
            if img_in.ndim != 2:
                _error(
                    channel_name,
                    f"{img_in.ndim}-dimensional images are not supported",
                )
            _image_validation(
                shape=img_in.shape,
                dtype=img_in.dtype,
                target_shape=target_shape,
                msg_tag=channel_name,
                is_mask=is_mask,
            )
    elif isinstance(input_data, list):
        in_imgs = []
        for i, path in tqdm(
            enumerate(input_data, 1),
            total=len(input_data),
            desc="Loading images",
            bar_format=TQDM_FORMAT,
        ):
            img_in = zarr.open(tifffile.imread(path, level=0, aszarr=True))
            if img_in.ndim == 2:
                shape = img_in.shape
                imgs = [img_in]
            elif img_in.ndim == 3:
                shape = img_in.shape[1:]
                imgs = [
                    zarr.open(tifffile.imread(path, key=i, level=0, aszarr=True))
                    for i in range(img_in.shape[0])
                ]
            else:
                _error(
                    path,
                    f"{img_in.ndim}-dimensional images are not supported",
                )
            if i == 1:
                target_shape = shape
            _image_validation(
                shape=shape,
                dtype=img_in.dtype,
                target_shape=target_shape,
                msg_tag=str(path),
                is_mask=is_mask,
            )
            in_imgs.extend(imgs)
    target_dtype = max([img.dtype for img in in_imgs])
    in_imgs = [img.astype(target_dtype) for img in in_imgs]

    num_channels = len(in_imgs)
    if channel_names and len(channel_names) != num_channels:
        _error(
            output_f,
            f"Number of channel names ({len(channel_names)}) does not"
            f" match number of channels in final image ({num_channels}).",
        )

    num_levels = np.ceil(np.log2(max(target_shape) / tile_size)) + 1
    num_levels = 1 if num_levels < 1 else int(num_levels)
    factors = 2 ** np.arange(num_levels)
    # shape of the pyramid
    shapes = np.ceil(np.array(target_shape) / factors[:, None]).astype(int)
    # shape of the tiles in the pyramid
    cshapes = np.ceil(shapes / tile_size).astype(int)

    metadata = {"UUID": uuid.uuid4().urn}
    if pixel_size:
        metadata.update(
            {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            }
        )
    if channel_names:
        metadata.update(
            {
                "Channel": {"Name": channel_names},
            }
        )

    pool = concurrent.futures.ThreadPoolExecutor(num_threads)

    def tiles0():
        """
        Generate tiles for the first level of the pyramid
        """
        ts = tile_size
        ch, cw = cshapes[0]
        for c, zimg in enumerate(in_imgs, 1):
            img = zimg[:]
            for j in range(ch):
                for i in range(cw):
                    tile = img[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                    yield tile
            del img

    def tiles(level):
        """
        Generate tiles for the given level of the pyramid
        """
        with tifffile.TiffFile(output_f, is_ome=False) as tiff_out:
            zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
            ts = tile_size * 2

        def tile(coords):
            """
            Generate a tile for the given coordinates
            """
            c, j, i = coords
            if zimg.ndim == 2:
                assert c == 0
                tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            else:
                tile = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            if is_mask:
                # Use nearest-neighbor downsampling for masks
                # Simply take every second pixel instead of averaging
                tile = tile[::2, ::2]
            else:
                tile = skimage.transform.downscale_local_mean(tile, (2, 2))
                tile = np.round(tile).astype(target_dtype)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(num_channels), range(ch), range(cw))
        yield from pool.map(tile, coords)

    with tifffile.TiffWriter(output_f, ome=True, bigtiff=True) as writer:
        for level, shape in tqdm(
            enumerate(shapes),
            total=len(shapes),
            desc="Writing images",
            bar_format=TQDM_FORMAT,
        ):
            if level == 0:
                writer.write(
                    data=tiles0(),
                    shape=(num_channels,) + tuple(shape),
                    subifds=num_levels - 1,
                    dtype=target_dtype,
                    tile=(tile_size, tile_size),
                    compression="adobe_deflate",
                    predictor=True,
                    metadata=metadata,
                )
            else:
                writer.write(
                    data=tiles(level),
                    shape=(num_channels,) + tuple(shape),
                    subfiletype=1,
                    dtype=target_dtype,
                    tile=(tile_size, tile_size),
                    compression="adobe_deflate",
                    predictor=True,
                )
    print()


def main():
    # Test 1: 3D pyramidal image
    ometiff_f = (
        "/mnt/nfs/home/wenruiwu/projects/pyqupath/data/ometiff/test_3d_pyramid.ome.tiff"
    )
    # temp = tifffile.imread(ometiff_f)
    # tifffile.imwrite(
    #     "/mnt/nfs/home/wenruiwu/projects/pyqupath/data/ometiff/test_3d_non_pyramid.ome.tiff",
    #     temp,
    #     metadata={
    #         "axes": "CYX",
    #         "Channel": {"Name": TiffReader.extract_channels_from_ometiff(ometiff_f)},
    #     },
    #     ome=True,
    # )

    tiff_reader = TiffZarrReader(ometiff_f)
    print(tiff_reader.channel_names)
    channels = ["DAPI", "CD45"]
    img = tiff_reader.zimg[tiff_reader.channel_index(channels), :100, :100]
    print(img.shape)

    img_dict = {
        channel_name: tiff_reader.zimg_dict[channel_name][:100, :100]
        for channel_name in channels
    }
    print({channel_name: img.shape for channel_name, img in img_dict.items()})

    # Test 2: 3D non-pyramidal image
    ometiff_f = Path(
        "/mnt/nfs/home/wenruiwu/projects/pyqupath/data/ometiff/test_3d_non_pyramid.ome.tiff"
    )
    tiff_reader = TiffZarrReader(ometiff_f)
    print(tiff_reader.channel_names)
    channels = ["DAPI", "CD45"]
    img = tiff_reader.zimg[tiff_reader.channel_index(channels), :100, :100]
    print(img.shape)

    img_dict = {
        channel_name: tiff_reader.zimg_dict[channel_name][:100, :100]
        for channel_name in channels
    }
    print({channel_name: img.shape for channel_name, img in img_dict.items()})
