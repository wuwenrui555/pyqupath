from __future__ import division, print_function

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import pathlib
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
import pandas as pd
import skimage.transform
import tifffile
import zarr
from tqdm import tqdm

###############################################################################
# pyramidal OME-TIFF writer
# https://github.com/labsyspharm/ome-tiff-pyramid-tools/blob/master/pyramid_assemble.py
###############################################################################


def format_shape(shape):
    return "%d x %d" % (shape[1], shape[0])


def error(path, msg):
    print(f"\nERROR: {path}: {msg}")
    sys.exit(1)


def pyramid_assemble(args=None):
    """
    Assemble a pyramidal OME-TIFF file.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments. If None, arguments are parsed from sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_paths",
        metavar="input.tif",
        type=pathlib.Path,
        nargs="+",
        help="List of TIFF files to combine. All images must have the same"
        " dimensions and pixel type. All pages of multi-page images will be"
        " included by default; the suffix ,p may be appended to the filename to"
        " specify a single page p.",
    )
    parser.add_argument(
        "out_path",
        metavar="output.ome.tif",
        type=pathlib.Path,
        help="Output filename. Script will exit immediately if file exists.",
    )
    parser.add_argument(
        "--pixel-size",
        metavar="MICRONS",
        type=float,
        default=None,
        help="Pixel size in microns. Will be recorded in OME-XML metadata.",
    )
    parser.add_argument(
        "--channel-names",
        metavar="CHANNEL",
        nargs="+",
        help="Channel names. Will be recorded in OME-XML metadata. Number of"
        " names must match number of channels in final output file.",
    )
    parser.add_argument(
        "--tile-size",
        metavar="PIXELS",
        type=int,
        default=1024,
        help="Width of pyramid tiles in output file (must be a multiple of 16);"
        " default is 1024",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        default=False,
        help="Adjust processing for label mask or binary mask images (currently"
        " just switch to nearest-neighbor downsampling)",
    )
    parser.add_argument(
        "--num-threads",
        metavar="N",
        type=int,
        default=0,
        help="Number of parallel threads to use for image downsampling; default"
        " is number of available CPUs",
    )
    args = parser.parse_args(args)
    in_paths = args.in_paths
    out_path = args.out_path
    is_mask = args.mask
    if out_path.exists():
        error(out_path, "Output file already exists, remove before continuing.")

    if args.num_threads == 0:
        if hasattr(os, "sched_getaffinity"):
            args.num_threads = len(os.sched_getaffinity(0))
        else:
            args.num_threads = multiprocessing.cpu_count()
        print(
            f"Using {args.num_threads} worker threads based on detected CPU" " count."
        )
        print()
    tifffile.TIFF.MAXWORKERS = args.num_threads
    tifffile.TIFF.MAXIOWORKERS = args.num_threads * 5

    in_imgs = []
    for i, path in tqdm(
        enumerate(in_paths, 1), total=len(in_paths), desc="Loading images"
    ):
        spath = str(path)
        if match := re.search(r",(\d+)$", spath):
            c = int(match.group(1))
            path = pathlib.Path(spath[: match.start()])
        else:
            c = None
        img_in = zarr.open(tifffile.imread(path, key=c, level=0, aszarr=True))
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
            error(
                path,
                f"{img_in.ndim}-dimensional images are not supported",
            )
        if i == 1:
            base_shape = shape
            dtype = img_in.dtype
            if dtype == np.uint32:
                if not is_mask:
                    error(
                        path,
                        "uint32 images are only supported in --mask mode."
                        " Please contact the authors if you need support for"
                        " intensity-based uint32 images.",
                    )
            elif dtype == np.uint16 or dtype == np.uint8:
                pass
            else:
                error(
                    path,
                    f"Can't handle dtype '{dtype}' yet, please contact the"
                    f" authors.",
                )
        else:
            if shape != base_shape:
                error(
                    path,
                    f"Expected shape {base_shape} to match first input image,"
                    f" got {shape} instead.",
                )
            if img_in.dtype != dtype:
                error(
                    path,
                    f"Expected dtype '{dtype}' to match first input image,"
                    f" got '{img_in.dtype}' instead.",
                )
        in_imgs.extend(imgs)

    num_channels = len(in_imgs)
    num_levels = np.ceil(np.log2(max(base_shape) / args.tile_size)) + 1
    factors = 2 ** np.arange(num_levels)
    shapes = np.ceil(np.array(base_shape) / factors[:, None]).astype(int)
    cshapes = np.ceil(shapes / args.tile_size).astype(int)

    if args.channel_names and len(args.channel_names) != num_channels:
        error(
            out_path,
            f"Number of channel names ({len(args.channel_names)}) does not"
            f" match number of channels in final image ({num_channels}).",
        )

    pool = concurrent.futures.ThreadPoolExecutor(args.num_threads)

    def tiles0():
        ts = args.tile_size
        ch, cw = cshapes[0]
        for c, zimg in enumerate(in_imgs, 1):
            img = zimg[:]
            for j in range(ch):
                for i in range(cw):
                    tile = img[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                    yield tile
            del img

    def tiles(level):
        tiff_out = tifffile.TiffFile(args.out_path, is_ome=False)
        zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
        ts = args.tile_size * 2

        def tile(coords):
            c, j, i = coords
            tile = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            tile = skimage.transform.downscale_local_mean(tile, (2, 2))
            tile = np.round(tile).astype(dtype)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(num_channels), range(ch), range(cw))
        yield from pool.map(tile, coords)

    metadata = {
        "UUID": uuid.uuid4().urn,
    }
    if args.pixel_size:
        metadata.update(
            {
                "PhysicalSizeX": args.pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": args.pixel_size,
                "PhysicalSizeYUnit": "µm",
            }
        )
    if args.channel_names:
        metadata.update(
            {
                "Channel": {"Name": args.channel_names},
            }
        )
    with tifffile.TiffWriter(args.out_path, ome=True, bigtiff=True) as writer:
        for level, shape in tqdm(
            enumerate(shapes), total=len(shapes), desc="Writing images"
        ):
            if level == 0:
                writer.write(
                    data=tiles0(),
                    shape=(num_channels,) + tuple(shapes[0]),
                    subifds=num_levels - 1,
                    dtype=dtype,
                    tile=(args.tile_size, args.tile_size),
                    compression="adobe_deflate",
                    predictor=True,
                    metadata=metadata,
                )
            else:
                writer.write(
                    data=tiles(level),
                    shape=(num_channels,) + tuple(shape),
                    subfiletype=1,
                    dtype=dtype,
                    tile=(args.tile_size, args.tile_size),
                    compression="adobe_deflate",
                    predictor=True,
                )
    print()


def pyramid_assemble_from_dict(
    im_dict: dict[str, np.ndarray],
    out_path: str,
    channel_names: list[str] = None,
    pixel_size: float = None,
    tile_size: int = 1024,
    num_threads: int = 0,
):
    """
    Assemble a pyramidal OME-TIFF file from a dictionary of images.

    Parameters
    ----------
    im_dict : dict of str to np.ndarray
        A dictionary where keys are channel names and values are 2D numpy arrays
        representing the images.
    channel_names : list of str
        Names of the channels in the OME-TIFF file. Each name corresponds to a
        channel in the `im_dict`. Default is None, meaning the channel names will
        be the keys of the `im_dict`.
    out_path : str
        Output filename. Script will exit immediately if file exists.
    pixel_size : float, optional
        Pixel size in microns. Will be recorded in OME-XML metadata.
    tile_size : int, optional
        Width of pyramid tiles in output file (must be a multiple of 16).
        Default is 1024.
    num_threads : int, optional
        Number of parallel threads to use for image downsampling. Default is
        number of available CPUs.
    """
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        error(out_path, "Output file already exists, remove before continuing.")

    if num_threads == 0:
        if hasattr(os, "sched_getaffinity"):
            num_threads = len(os.sched_getaffinity(0))
        else:
            num_threads = multiprocessing.cpu_count()
        print(f"Using {num_threads} worker threads based on detected CPU" " count.")
        print()
    tifffile.TIFF.MAXWORKERS = num_threads
    tifffile.TIFF.MAXIOWORKERS = num_threads * 5

    if channel_names is None:
        channel_names = list(im_dict.keys())
    in_imgs = [im_dict[name] for name in channel_names]

    # ensure the shape and dtype of the images are the same
    base_shape = next(iter(im_dict.values())).shape
    dtype = next(iter(im_dict.values())).dtype
    for name, im in im_dict.items():
        if im.shape != base_shape:
            error(
                name,
                f"Expected shape {base_shape} to match first input image,"
                f" got {im.shape} instead.",
            )
        if im.dtype != dtype:
            error(
                name,
                f"Expected dtype '{dtype}' to match first input image,"
                f" got '{im.dtype}' instead.",
            )

    # calculate the pyramid levels
    num_channels = len(in_imgs)
    num_levels = np.ceil(np.log2(max(base_shape) / tile_size)) + 1
    factors = 2 ** np.arange(num_levels)
    shapes = np.ceil(np.array(base_shape) / factors[:, None]).astype(int)
    cshapes = np.ceil(shapes / tile_size).astype(int)

    if channel_names and len(channel_names) != num_channels:
        error(
            out_path,
            f"Number of channel names ({len(channel_names)}) does not"
            f" match number of channels in final image ({num_channels}).",
        )

    pool = concurrent.futures.ThreadPoolExecutor(num_threads)

    def tiles0():
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
        tiff_out = tifffile.TiffFile(out_path, is_ome=False)
        zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
        ts = tile_size * 2

        def tile(coords):
            c, j, i = coords
            tile = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            tile = skimage.transform.downscale_local_mean(tile, (2, 2))
            tile = np.round(tile).astype(dtype)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(num_channels), range(ch), range(cw))
        yield from pool.map(tile, coords)

    metadata = {
        "UUID": uuid.uuid4().urn,
    }
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
    with tifffile.TiffWriter(out_path, ome=True, bigtiff=True) as writer:
        for level, shape in tqdm(
            enumerate(shapes), total=len(shapes), desc="Writing images"
        ):
            if level == 0:
                writer.write(
                    data=tiles0(),
                    shape=(num_channels,) + tuple(shapes[0]),
                    subifds=num_levels - 1,
                    dtype=dtype,
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
                    dtype=dtype,
                    tile=(tile_size, tile_size),
                    compression="adobe_deflate",
                    predictor=True,
                )
    print()


def export_ometiff_pyramid(
    paths_tiff: list[str],
    path_ometiff: str,
    channel_names: list[str],
    tile_size: int = 256,
    num_threads: int = 20,
):
    """
    Generate a pyramidal OME-TIFF file from multiple input TIFF files.

    This function combines multiple input TIFF files into a single multi-channel
    OME-TIFF file, adds channel metadata, and builds a multi-resolution pyramid
    for efficient visualization and processing.
    ( https://github.com/labsyspharm/ome-tiff-pyramid-tools)

    Parameters
    ----------
    paths_tiff : list of str
        A list of file paths to the input TIFF images. Each file corresponds
        to a specific channel in the final OME-TIFF.
    path_ometiff : str
        Path to the output OME-TIFF file. If the file already exists, the process
        will terminate to prevent overwriting.
    channel_names : list of str
        Names of the channels in the OME-TIFF file. Each name corresponds to a
        channel in the `im_dict`. Default is None, meaning the channel names will
        be the keys of the `im_dict`.
    tile_size : int, optional, default=256
        The width and height of tiles in the pyramidal TIFF. Smaller tile sizes
        can improve performance in certain scenarios.
    num_threads : int, optional, default=20
        The number of threads to use for downsampling images and constructing the
        pyramid. Higher values can improve performance on multi-core systems.
    """
    args = [
        *paths_tiff,
        path_ometiff,
        "--channel-names",
        *channel_names,
        "--tile-size",
        str(tile_size),
        "--num-threads",
        str(num_threads),
    ]
    pyramid_assemble(args)


def export_ometiff_pyramid_from_dict(
    im_dict: dict[str, np.ndarray],
    path_ometiff: str,
    channel_names: list[str] = None,
    tile_size: int = 256,
    num_threads: int = 20,
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
        Names of the channels in the OME-TIFF file. Each name corresponds to a TIFF
        file in `input_tiff_paths`. The length of this list must match the number
        of files in `input_tiff_paths`.
    tile_size : int, optional, default=256
        The width and height of tiles in the pyramidal TIFF. Smaller tile sizes
        can improve performance in certain scenarios.
    n_threads : int, optional, default=20
        The number of threads to use for downsampling images and constructing the
        pyramid. Higher values can improve performance on multi-core systems.
    """
    pyramid_assemble_from_dict(
        im_dict=im_dict,
        out_path=path_ometiff,
        channel_names=channel_names,
        tile_size=tile_size,
        num_threads=num_threads,
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
    export_ometiff_pyramid_from_dict(im_dict, path_ometiff, markers_name)


###############################################################################
# metadata extraction
###############################################################################


def parse_xml_string_ometiff(xml_string):
    """
    Parse an XML string from an OME-TIFF file to extract metadata.

    Parameters
    ----------
    xml_string : str
        The XML string containing the OME-TIFF metadata.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing channel metadata.
    """
    root = ET.fromstring(xml_string)
    channels = root.findall(".//{*}Channel")
    metadata = pd.DataFrame([channel.attrib for channel in channels])
    return metadata


def parse_xml_string_qptiff(xml_string):
    """
    Parse an XML string from a QPTIFF file to extract metadata.

    Parameters
    ----------
    xml_string : str
        The XML string containing the QPTIFF metadata.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing metadata.
    """
    root = ET.fromstring(xml_string)

    scan_profile = root.find(".//ScanProfile")
    if scan_profile is None:
        raise ValueError("ScanProfile element not found in the provided XML string.")

    scan_profile_data = json.loads(scan_profile.text)
    wells = scan_profile_data.get("experimentDescription").get("wells")
    metadata = pd.concat(
        [
            pd.DataFrame(well.get("items")).assign(wellName=well.get("wellName"))
            for well in wells
        ],
        ignore_index=True,
    )
    return metadata


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
    with tifffile.TiffFile(path_ometiff) as im:
        xml_string = im.ome_metadata
    metadata = parse_xml_string_ometiff(xml_string)
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
        series = im.series[0]
        xml_string = series.pages[0].tags["ImageDescription"].value
        metadata = parse_xml_string_qptiff(xml_string)
        channels = (
            metadata.loc[
                (metadata["markerName"] != "--") & (metadata["panel"] != "Inventoried")
            ]
            .drop_duplicates(["id", "markerName"])["markerName"]
            .tolist()
        )
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
    filetype,
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
    # Step 1: Extract channel names
    if path_markerlist is None:
        if filetype == "qptiff":
            channels_name = extract_channels_from_qptiff(path_tiff)
        elif filetype == "ome.tiff":
            channels_name = extract_channels_from_ometiff(path_tiff)
        else:
            raise ValueError("Filetype must be either 'qptiff' or 'ome.tiff'")
    else:
        channels_name = np.loadtxt(path_markerlist, dtype=str).tolist()

    # Default to loading in original order if `channels_order` is not specified
    if channels_order is None:
        channels_order = channels_name

    # Step 2: Validate channels_order against channels_name
    missing_markers = set(channels_order) - set(channels_name)
    if missing_markers:
        raise ValueError(
            f"The following markers are not found in the TIFF file: {missing_markers}"
        )

    # Step 3: Validate channels_rename if provided
    if channels_rename:
        if len(channels_rename) != len(channels_order):
            raise ValueError(
                "The length of `channels_rename` must match `channels_order`."
            )

    # Step 4: Load the image data
    ## All channels are requested, no reordering needed
    if set(channels_name) == set(channels_order):
        im = tifffile.imread(path_tiff)  # faster than using generator
        im_dict = OrderedDict((channels_name[i], im[i]) for i in range(im.shape[0]))
        if channels_name != channels_order:
            im_dict = OrderedDict((name, im_dict[name]) for name in channels_order)
    ## Only a subset of channels is requested
    else:
        index = [channels_name.index(channel) for channel in channels_order]
        im_generator = tiff_highest_resolution_generator(
            path_tiff, asarray=True, index=index
        )
        names = channels_order if channels_rename is None else channels_rename
        im_dict = OrderedDict(
            (names[i], im)
            for i, im in tqdm(
                enumerate(im_generator), total=len(index), desc="Loading images"
            )
        )

    return im_dict
