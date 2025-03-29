# %%
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import pathlib
import re
import uuid
from pathlib import Path
from typing import Union
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import skimage.transform
import tifffile
import zarr
from tqdm import tqdm

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
        self.zimg = zarr.open(tifffile.imread(self.path, level=0, aszarr=True))

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
                tifffile.imread(self.path, key=i, level=0, aszarr=True)
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


def _validate_image_2d(
    shape: tuple[int, int],
    dtype: np.dtype,
    target_shape: tuple[int, int],
    is_mask: bool = False,
    msg_tag: str = None,
):
    """
    Validate 2D image shape and data type.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the image to validate.
    dtype : np.dtype
        Data type of the image.
    target_shape : tuple[int, int]
        Expected shape of the image.
    is_mask : bool, optional
        Whether the image is a mask. Default is False.
    msg_tag : str, optional
        Tag for error messages. Default is None.

    Raises
    ------
    ValueError
        When validation fails.
    """
    if msg_tag is not None:
        msg_tag = f"{msg_tag}: "
    if dtype == np.uint32 or dtype == np.int32:
        if not is_mask:
            msg = f"{msg_tag}32-bit images are only supported in is_mask = True"
            raise ValueError(msg)
    elif dtype not in (np.uint8, np.uint16):
        msg = f"{msg_tag}Unsupported dtype: {dtype}"
        raise ValueError(msg)

    if shape != target_shape:
        msg = f"{msg_tag}Shape mismatch: expected {target_shape}, got {shape}"
        raise ValueError(msg)


def _process_dict_input(
    input_data: dict[str, Union[np.ndarray, zarr.Array]],
    channel_names: list[str] = None,
    is_mask: bool = False,
) -> tuple[list[zarr.Array], list[str], tuple[int, int], np.dtype]:
    """
    Process dictionary input format.

    Parameters
    ----------
    input_data : dict[str, Union[np.ndarray, zarr.Array]]
        Dictionary where keys are channel names and values are numpy arrays or
        zarr arrays.
    channel_names : list[str], optional
        Names of the channels. If None, uses dictionary keys.
    is_mask : bool, optional
        Whether the images are masks.

    Returns
    -------
    tuple
        (in_imgs, channel_names, target_shape, target_dtype)
    """
    # Channel names
    if channel_names is None:
        channel_names = list(input_data.keys())
    if len(channel_names) != len(input_data):
        raise ValueError(
            f"channel_names: Expected {len(input_data)} channel names, got {len(channel_names)}"
        )

    # Target image shape
    target_shape = next(iter(input_data.values())).shape[-2:]

    # Add validated images to list
    in_imgs = []
    in_chns = []
    for channel_name, img_in in zip(channel_names, input_data.values()):
        if isinstance(img_in, (np.ndarray, zarr.Array)):
            if img_in.ndim == 2:
                _validate_image_2d(
                    shape=img_in.shape,
                    dtype=img_in.dtype,
                    target_shape=target_shape,
                    is_mask=is_mask,
                    msg_tag=channel_name,
                )
                in_imgs.append(zarr.array(img_in))
                in_chns.append(channel_name)
            elif img_in.ndim == 3:
                for i in range(img_in.shape[0]):
                    img = img_in[i]
                    _validate_image_2d(
                        shape=img.shape,
                        dtype=img.dtype,
                        target_shape=target_shape,
                        is_mask=is_mask,
                        msg_tag=f"{channel_name}_{i}",
                    )
                    in_imgs.append(zarr.array(img))
                    in_chns.append(f"{channel_name}_{i}")
            else:
                raise ValueError(
                    f"{channel_name}: Unsupported dimensions: {img_in.ndim}"
                )
        else:
            raise ValueError(f"{channel_name}: Unsupported type: {type(img_in)}")

    # Convert to uniform dtype
    target_dtype = max([img.dtype for img in in_imgs])
    in_imgs = [img.astype(target_dtype) for img in in_imgs]

    return in_imgs, in_chns, target_shape, target_dtype


def _process_array_input(
    input_data: Union[np.ndarray, zarr.Array],
    channel_names: list[str] = None,
    is_mask: bool = False,
) -> tuple[list[zarr.Array], list[str], tuple[int, int], np.dtype]:
    """
    Process array input format (2D or 3D).

    Parameters
    ----------
    input_data : Union[np.ndarray, zarr.Array]
        2D or 3D array (C, H, W) input.
    channel_names : list[str], optional
        Names of the channels.
    is_mask : bool, optional
        Whether the images are masks.

    Returns
    -------
    tuple
        (in_imgs, channel_names, target_shape, target_dtype)
    """
    # Convert 2D array to 3D array
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, ...]
    elif input_data.ndim == 3:
        pass
    else:
        raise ValueError(
            f"input_data: Expected 2D or 3D array, got shape {input_data.shape}"
        )

    # Channel names and number of channels
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(input_data.shape[0])]
    if len(channel_names) != input_data.shape[0]:
        raise ValueError(
            f"channel_names: Expected {input_data.shape[0]} channel names, got {len(channel_names)}"
        )

    # Target image shape
    target_shape = input_data.shape[-2:]

    # Add validated images to list
    in_imgs = []
    in_chns = []
    for channel_name, img in zip(channel_names, input_data):
        _validate_image_2d(
            shape=img.shape,
            dtype=img.dtype,
            target_shape=target_shape,
            is_mask=is_mask,
            msg_tag=channel_name,
        )
        in_imgs.append(zarr.array(img))
        in_chns.append(channel_name)

    # Convert to uniform dtype
    target_dtype = max([img.dtype for img in in_imgs])
    in_imgs = [img.astype(target_dtype) for img in in_imgs]

    return in_imgs, in_chns, target_shape, target_dtype


def _process_file_input(
    input_data: list[Union[str, pathlib.Path]],
    channel_names: list[str] = None,
    is_mask: bool = False,
) -> tuple[list[zarr.Array], list[str], tuple[int, int], np.dtype]:
    """
    Process file paths input format.

    Parameters
    ----------
    input_data : list[Union[str, pathlib.Path]]
        List of file paths to TIFF images.
    channel_names : list[str], optional
        Names of the channels.
    is_mask : bool, optional
        Whether the images are masks.

    Returns
    -------
    tuple
        (in_imgs, channel_names, target_shape, target_dtype)
    """
    # Channel names
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(len(input_data))]
    if len(channel_names) != len(input_data):
        raise ValueError(
            f"channel_names: Expected {len(input_data)} channel names, got {len(channel_names)}"
        )

    # Add validated images to list
    in_imgs = []
    in_chns = []
    for i, path in tqdm(
        enumerate(input_data),
        total=len(input_data),
        desc="Loading images",
        bar_format=TQDM_FORMAT,
    ):
        channel_name = channel_names[i]
        img_in = zarr.open(tifffile.imread(path, level=0, aszarr=True))
        if i == 0:
            target_shape = img_in.shape[-2:]

        if img_in.ndim == 2:
            shape = img_in.shape
            _validate_image_2d(
                shape=shape,
                dtype=img_in.dtype,
                target_shape=target_shape,
                is_mask=is_mask,
                msg_tag=channel_name,
            )
            in_imgs.append(zarr.array(img_in))
            in_chns.append(channel_name)
        elif img_in.ndim == 3:
            shape = img_in.shape[1:]
            for j in range(img_in.shape[0]):
                img = img_in[j]
                _validate_image_2d(
                    shape=img.shape,
                    dtype=img.dtype,
                    target_shape=target_shape,
                    is_mask=is_mask,
                    msg_tag=f"{channel_name}_{j}",
                )
                in_imgs.append(zarr.array(img))
                in_chns.append(f"{channel_name}_{j}")
        else:
            raise ValueError(f"{path}: Unsupported dimensions: {img_in.ndim}")

    # Convert to uniform dtype
    target_dtype = max([img.dtype for img in in_imgs])
    in_imgs = [img.astype(target_dtype) for img in in_imgs]

    return in_imgs, in_chns, target_shape, target_dtype


def _process_input_data(
    input_data: Union[
        list[Union[str, pathlib.Path]],
        dict[str, Union[np.ndarray, zarr.Array]],
        np.ndarray,
        zarr.Array,
    ],
    channel_names: list[str] = None,
    is_mask: bool = False,
) -> tuple[list[zarr.Array], list[str], tuple[int, int], np.dtype]:
    """
    Process input data into a unified format.

    Parameters
    ----------
    input_data : Union[list[Union[str, pathlib.Path]], dict[str, Union[np.ndarray, zarr.Array]], np.ndarray, zarr.Array]
        Input data in one of four formats:
        1. A list of file paths to TIFF images
        2. A dictionary where keys are channel names and values are numpy arrays or zarr arrays
        3. A 2D numpy array or zarr array (will be treated as single channel)
        4. A 3D numpy array or zarr array with shape (C, H, W) where C is the number of channels
    channel_names : list[str], optional
        Names of the channels in the OME-TIFF file. Each name corresponds to a
        channel in the `input_data`. The length of this list must match the number
        of channels. Default is None.
    is_mask : bool, optional
        Whether the input data represents mask images. This affects validation
        rules for data types. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - list[zarr.Array]: List of zarr arrays
        - list[str]: List of channel names
        - tuple[int, int]: Target shape of the images
        - np.dtype: Target data type for all images
    """
    if isinstance(input_data, dict):
        return _process_dict_input(input_data, channel_names, is_mask)
    elif isinstance(input_data, (np.ndarray, zarr.Array)):
        return _process_array_input(input_data, channel_names, is_mask)
    else:
        return _process_file_input(input_data, channel_names, is_mask)


def _calculate_pyramid_levels(
    target_shape: tuple[int, int], tile_size: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Calculate pyramid levels and shapes.

    Parameters
    ----------
    target_shape : tuple[int, int]
        The shape of the base image (height, width).
    tile_size : int
        The size of tiles in the pyramid.

    Returns
    -------
    tuple
        A tuple containing:
        - int: Number of pyramid levels
        - np.ndarray: Array of shapes of the pyramid
        - np.ndarray: Array of shapes of the tiles in the pyramid
    """
    num_levels = max(1, int(np.ceil(np.log2(max(target_shape) / tile_size)) + 1))
    factors = 2 ** np.arange(num_levels)
    shapes = np.ceil(np.array(target_shape) / factors[:, None]).astype(int)
    cshapes = np.ceil(shapes / tile_size).astype(int)
    return num_levels, shapes, cshapes


def _create_metadata(pixel_size: float = None, channel_names: list[str] = None) -> dict:
    """
    Create OME-TIFF metadata.

    Parameters
    ----------
    pixel_size : float, optional
        Physical size of pixels in microns. Will be recorded in OME-XML metadata.
    channel_names : list[str], optional
        Names of the channels in the OME-TIFF file.

    Returns
    -------
    dict
        Dictionary containing OME-TIFF metadata including UUID, physical size,
        and channel names if provided.
    """
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
    return metadata


def _create_tile_generators(
    in_imgs: list[zarr.Array],
    cshapes: np.ndarray,
    tile_size: int,
    num_channels: int,
    is_mask: bool,
    target_dtype: np.dtype,
    num_threads: int,
    output_f: pathlib.Path,
) -> tuple[callable, callable]:
    """Create tile generators for base and pyramid levels.

    Parameters
    ----------
    in_imgs : list[zarr.Array]
        List of input images as zarr arrays.
    cshapes : np.ndarray
        Array of shapes of the tiles in the pyramid.
    tile_size : int
        Size of tiles in the pyramid.
    num_channels : int
        Number of channels in the images.
    is_mask : bool
        Whether the images are masks (affects downsampling method).
    target_dtype : np.dtype
        Target data type for all tiles.
    num_threads : int
        Number of threads for parallel processing.
    output_f : pathlib.Path
        Path to the output file.

    Returns
    -------
    tuple[callable, callable]
        A tuple containing two generator functions:
        - First function generates tiles for the base level
        - Second function generates tiles for pyramid levels
    """
    pool = concurrent.futures.ThreadPoolExecutor(num_threads)

    def tiles0():
        """Generate tiles for the first level."""
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
        """Generate tiles for pyramid levels."""
        with tifffile.TiffFile(output_f, is_ome=False) as tiff_out:
            zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
            ts = tile_size * 2

        def tile(coords):
            c, j, i = coords
            if zimg.ndim == 2:
                assert c == 0
                tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            else:
                tile = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            if is_mask:
                tile = tile[::2, ::2]
            else:
                tile = skimage.transform.downscale_local_mean(tile, (2, 2))
                tile = np.round(tile).astype(target_dtype)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(num_channels), range(ch), range(cw))
        yield from pool.map(tile, coords)

    return tiles0, tiles


def export_ometiff_pyramid(
    input_data: Union[list[Union[str, pathlib.Path]], dict[str, np.ndarray]],
    output_f: Union[str, pathlib.Path],
    channel_names: list[str] = None,
    pixel_size: float = None,
    tile_size: int = 256,
    is_mask: bool = False,
    num_threads: int = 8,
    overwrite: bool = True,
):
    """
    Assemble a pyramidal OME-TIFF file.

    Parameters
    ----------
    input_data : Union[list[Union[str, pathlib.Path]], dict[str, np.ndarray]]
        A list of file paths to the input TIFF images or a dictionary where keys
        are channel names and values are 2D numpy arrays representing the images.
        All images must have the same dimensions and pixel type.
    output_f : Union[str, pathlib.Path]
        Path to the output OME-TIFF file.
    channel_names : list[str], optional
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
        switch to nearest-neighbor downsampling). Default is False.
    num_threads : int, optional
        Number of parallel threads to use for image downsampling. Default is
        number of available CPUs.
    overwrite : bool, optional
        If True, the function will overwrite the output file if it already exists.
        If False, the function will terminate to prevent overwriting. Default
        is True.

    Raises
    ------
    ValueError
        When input validation fails.
    FileExistsError
        When output file exists and overwrite is False.
    """
    # Setup output file
    output_f = pathlib.Path(output_f)
    if output_f.exists():
        if overwrite:
            print(f"Overwriting existing file: {output_f}")
            output_f.unlink()
        else:
            raise FileExistsError(f"Output file already exists: {output_f}")

    # Setup threads
    if num_threads == 0:
        if hasattr(os, "sched_getaffinity"):
            num_threads = len(os.sched_getaffinity(0))
        else:
            num_threads = multiprocessing.cpu_count()
        print(f"Using {num_threads} worker threads")
    tifffile.TIFF.MAXWORKERS = num_threads
    tifffile.TIFF.MAXIOWORKERS = num_threads * 5

    # Process input data
    in_imgs, in_chns, target_shape, target_dtype = _process_input_data(
        input_data=input_data,
        channel_names=channel_names,
        is_mask=is_mask,
    )
    num_channels = len(in_chns)

    # Calculate pyramid levels
    num_levels, shapes, cshapes = _calculate_pyramid_levels(
        target_shape=target_shape,
        tile_size=tile_size,
    )

    # Create metadata
    metadata = _create_metadata(
        pixel_size=pixel_size,
        channel_names=in_chns,
    )

    # Create tile generators
    tiles0, tiles = _create_tile_generators(
        in_imgs=in_imgs,
        cshapes=cshapes,
        tile_size=tile_size,
        num_channels=num_channels,
        is_mask=is_mask,
        target_dtype=target_dtype,
        num_threads=num_threads,
        output_f=output_f,
    )

    # Write pyramid
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
