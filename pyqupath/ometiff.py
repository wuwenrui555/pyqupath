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
from collections import OrderedDict
from typing import Union
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import skimage.transform
import tifffile
import zarr
from tqdm import tqdm

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


###############################################################################
# pyramidal OME-TIFF writer
# https://github.com/labsyspharm/ome-tiff-pyramid-tools/blob/master/pyramid_assemble.py
###############################################################################


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


def export_ometiff_pyramid_from_dict(
    im_dict: dict[str, np.ndarray],
    path_ometiff: str,
    channel_names: list[str] = None,
    pixel_size: float = None,
    tile_size: int = 256,
    num_threads: int = 8,
    overwrite: bool = True,
) -> str:
    """
    Generate a pyramidal OME-TIFF file from a dictionary of images.

    Parameters
    ----------
    im_dict : dict of str to np.ndarray
        A dictionary where keys are channel names and values are 2D numpy arrays
        representing the images.
    path_ometiff : str
        Path to the output OME-TIFF file. If the file already exists, the process
        will terminate to prevent overwriting.
    channel_names : list of str
        Names of the channels in the OME-TIFF file. Each name corresponds to a
        channel in the `im_dict`. The order of the `channel_names` determines
        the order of the channels in the OME-TIFF file. Default is None, meaning
        the `channel_names` will be the keys of the `im_dict`.
    pixel_size : float, optional
        Pixel size in microns. Will be recorded in OME-XML metadata.
    tile_size : int, optional, default=256
        The width and height of tiles in the pyramidal TIFF. Smaller tile sizes
        can improve performance in certain scenarios.
    num_threads : int, optional, default=8
        The number of threads to use for downsampling images and constructing the
        pyramid. Higher values can improve performance on multi-core systems.
    overwrite : bool, optional, default=True
        If True, the function will overwrite the output file if it already exists.
        If False, the function will terminate to prevent overwriting.
    """
    export_ometiff_pyramid(
        input_data=im_dict,
        output_f=path_ometiff,
        channel_names=channel_names,
        pixel_size=pixel_size,
        tile_size=tile_size,
        num_threads=num_threads,
        overwrite=overwrite,
    )


def export_ometiff_pyramid_from_qptiff(
    path_qptiff: str,
    path_ometiff: str,
    path_markerlist: str = None,
):
    """
    Generate a pyramidal OME-TIFF file from a QPTIFF file.

    This function converts a QPTIFF file into an OME-TIFF pyramid format,
    extracting marker information either from the QPTIFF file itself or
    from a provided marker list file.

    Parameters
    ----------
    path_qptiff : str
        Path to the input QPTIFF file.
    path_ometiff : str
        Path to save the output OME-TIFF file.
    path_markerlist : str, optional
        Path to a marker list file. If None (default), marker names will be
        extracted directly from the QPTIFF file.
    """
    if not pathlib.Path(path_qptiff).exists():
        raise FileNotFoundError(f"QPTIFF not found: {path_qptiff}")

    # Extract marker names
    if path_markerlist is None:
        markers_name = extract_channels_from_qptiff(path_qptiff)
    else:
        if not pathlib.Path(path_markerlist).exists():
            raise FileNotFoundError(f"Marker list file not found: {path_markerlist}")
        else:
            markers_name = np.loadtxt(path_markerlist, dtype=str).tolist()

    # Read QPTIFF file and organize data
    im = tifffile.imread(path_qptiff)
    im_dict = OrderedDict((markers_name[i], im[i]) for i in range(im.shape[0]))

    # Export OME-TIFF pyramid
    pathlib.Path(path_ometiff).parent.mkdir(parents=True, exist_ok=True)
    export_ometiff_pyramid(
        input_data=im_dict,
        output_f=path_ometiff,
        channel_names=markers_name,
    )


###############################################################################
# channel extraction
###############################################################################


def extract_channels_from_ometiff(path_ometiff: str) -> list[str]:
    """
    Extract channel names from an OME-TIFF file.

    Parameters
    ----------
    path_ometiff : str
        Path to the OME-TIFF file.

    Returns
    -------
    list of str
        A list of channel names.
    """
    with tifffile.TiffFile(path_ometiff) as tif:
        ome_metadata = ElementTree.fromstring(tif.ome_metadata)
        ome_channels = ome_metadata.findall(".//{*}Channel")
        metadata = pd.DataFrame([channel.attrib for channel in ome_channels])
        channels = metadata["Name"].tolist()
    return channels


def extract_channels_from_qptiff(path_qptiff: str) -> list[str]:
    """
    Extract channel names from a QPTIFF file.

    Parameters
    ----------
    path_ometiff : str
        Path to the OME-TIFF file.

    Returns
    -------
    list of str
        A list of channel names.
    """
    with tifffile.TiffFile(path_qptiff) as im:
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
                pd.DataFrame(well.get("items")).assign(wellName=well.get("wellName"))
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


def extract_channels_from_qptiff_raw(path_qptiff: str) -> list[str]:
    """
    Extract channel names from a raw QPTIFF file.

    Parameters
    ----------
    path_ometiff : str
        Path to the OME-TIFF file.

    Returns
    -------
    list of str
        A list of channel names.
    """
    channels = []
    with tifffile.TiffFile(path_qptiff) as tif:
        for page in tif.series[0].pages:
            channel = ElementTree.fromstring(page.description).find("Name").text
            channels.append(channel)
    return channels


###############################################################################
# tiff reader
###############################################################################


def tiff_highest_resolution_generator(
    path: str, asarray: bool = False, index: list[int] = None
):
    """
    Generator to read the highest resolution level of a multi-page TIFF file.

    This function processes only the highest resolution level (first series)
    of a multi-page TIFF file, yielding each page (or frame) as a NumPy array.

    Parameters
    ----------
    path : str
        Path to the TIFF file.
    asarray : bool, optional
        If True, the generator yields each page as a NumPy array. If False,
        the generator yields each page as a TiffPage object. Default is False.
    index : list[int], optional
        List of indices of the pages to process. If None, all pages are processed.

    Yields
    ------
    numpy.ndarray
        A NumPy array representing each page (or frame) in the highest resolution level.
    """
    with tifffile.TiffFile(path) as tif:
        # Access the first series (highest resolution)
        series = tif.series[0]

        # Pages to process
        if index is None:
            pages = series.pages
        else:
            pages = series.pages[index]

        # Yield each page
        for page in pages:
            if asarray:
                yield page.asarray()
            else:
                yield page


def load_tiff_to_dict(
    path_tiff,
    filetype: str = None,
    channels_order: list[str] = None,
    channels_rename: list[str] = None,
    path_markerlist: str = None,
) -> OrderedDict[str, np.ndarray]:
    """
    Load a multi-channel TIFF file into a dictionary of channel images.

    Parameters
    ----------
    path_tiff : str
        Path to the TIFF file.
    filetype : str
        Filetype of the TIFF file. Must be either "qptiff" or "ome.tiff".
    channels_order : list[str], optional
        List of channel names in the desired order. If None, the channels will
        be loaded in the order they appear in the TIFF file or the marker list
        file (if provided). Default is None.
    channels_rename : list[str], optional
        List of new channel names. If provided, the channel names will be
        renamed according to this list. The length of `channels_rename` must
        match `channels_order`. Default is None.
    path_markerlist : str, optional
        Path to the marker list file. If provided, the channel names will be
        extracted from the marker list file. Default is None.

    Returns
    -------
    OrderedDict[str, np.ndarray]
        An ordered dictionary where keys are channel names (renamed if
        `channels_rename` is provided) and values are corresponding image arrays.
    """
    # Validate filetype
    if filetype is None:
        filetype = path_tiff.split(".", 1)[-1]
    if filetype not in ["qptiff", "ome.tiff"]:
        raise ValueError("Filetype must be either 'qptiff' or 'ome.tiff'")

    # Extract channel names
    if path_markerlist is None:
        if filetype == "qptiff":
            channels_name = extract_channels_from_qptiff(path_tiff)
        elif filetype == "ome.tiff":
            channels_name = extract_channels_from_ometiff(path_tiff)
        else:
            raise ValueError("Filetype must be either 'qptiff' or 'ome.tiff'")
    else:
        with open(path_markerlist) as f:
            channels_name = f.readlines()
            channels_name = [x.strip() for x in channels_name]

    # Default to loading in original order if `channels_order` is not specified
    if channels_order is None:
        channels_order = channels_name

    # Validate channels_order against channels_name
    missing_markers = set(channels_order) - set(channels_name)
    if missing_markers:
        raise ValueError(
            f"The following markers are not found in the TIFF file: {missing_markers}"
        )

    # Validate channels_rename if provided
    if channels_rename:
        if len(channels_rename) != len(channels_order):
            raise ValueError(
                "The length of `channels_rename` must match `channels_order`."
            )

    # Load the image data
    index = [channels_name.index(channel) for channel in channels_order]
    im_generator = tiff_highest_resolution_generator(
        path_tiff, asarray=True, index=index
    )
    names = channels_order if channels_rename is None else channels_rename
    im_dict = OrderedDict(
        (names[i], im)
        for i, im in tqdm(
            enumerate(im_generator),
            total=len(index),
            desc="Loading images",
            bar_format=TQDM_FORMAT,
        )
    )

    return im_dict


# %%
def main():
    """
    Test the tiff module.
    """
    img_dict = {"test": np.ones((1000, 1000), dtype=np.uint8)}
    export_ometiff_pyramid_from_dict(img_dict, "test.ome.tiff", channel_names=["test"])


if __name__ == "__main__":
    main()
