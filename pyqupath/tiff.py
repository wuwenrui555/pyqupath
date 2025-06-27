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
        tiff_f: Union[str, pathlib.Path],
        channel_names: list[str] = None,
    ):
        """
        Initialize the TIFF reader.

        Parameters
        ----------
        tiff_f : str or pathlib.Path
            Path to the TIFF file
        channel_names : list[str], optional
            Names of the channels. If None, will be inferred from the file type.
        """
        # Validate tiff_f
        tiff_f = pathlib.Path(tiff_f)
        if not tiff_f.exists():
            raise FileNotFoundError(f"File not found: {tiff_f}")

        # Initialize zarr reader
        self.zimg = zarr.open(tifffile.imread(tiff_f, level=0, aszarr=True))

        # Get channel names
        if self.zimg.ndim == 3:
            if channel_names is None:
                channel_names = [f"channel_{i}" for i in range(self.zimg.shape[0])]
            n_channel = self.zimg.shape[0]
        elif self.zimg.ndim == 2:
            if channel_names is None:
                channel_names = ["channel_0"]
            n_channel = 1
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.zimg.ndim}")
        if len(channel_names) != n_channel:
            raise ValueError(
                f"channel_names: Expected {n_channel} channel names, got {len(channel_names)}"
            )
        self.channel_names = channel_names

        # Generate zimg_dict with channel names as keys
        self.zimg_dict = {
            channel_name: zarr.open(
                tifffile.imread(tiff_f, key=i, level=0, aszarr=True)
            )
            for i, channel_name in enumerate(self.channel_names)
        }

    @classmethod
    def from_ometiff(
        cls,
        tiff_f: Union[str, pathlib.Path],
        markerlist_f: Union[str, pathlib.Path] = None,
    ) -> "TiffZarrReader":
        """
        Initialize a TiffZarrReader from an OME-TIFF file.
        """
        # Extract channel names
        if markerlist_f is None:
            channel_names = cls.extract_channel_names_ometiff(tiff_f)
        else:
            with open(markerlist_f) as f:
                channel_names = f.readlines()
                channel_names = [x.strip() for x in channel_names]
        return cls(tiff_f, channel_names)

    @classmethod
    def from_qptiff(
        cls,
        tiff_f: Union[str, pathlib.Path],
        markerlist_f: Union[str, pathlib.Path] = None,
    ) -> "TiffZarrReader":
        """
        Initialize a TiffZarrReader from a QPTIFF file.
        """
        # Extract channel names
        if markerlist_f is None:
            channel_names = cls.extract_channel_names_qptiff(tiff_f)
        else:
            with open(markerlist_f) as f:
                channel_names = f.readlines()
                channel_names = [x.strip() for x in channel_names]
        return cls(tiff_f, channel_names)

    @staticmethod
    def extract_channel_names_ometiff(path: Union[str, pathlib.Path]) -> list[str]:
        """
        Extract channel names from an OME-TIFF file.
        """
        with tifffile.TiffFile(path) as tif:
            ome_metadata = ElementTree.fromstring(tif.ome_metadata)
            ome_channels = ome_metadata.findall(".//{*}Channel")
            metadata = pd.DataFrame([channel.attrib for channel in ome_channels])
            if "Name" in metadata.columns:
                channel_names = metadata["Name"].tolist()
            else:
                channel_names = [f"Channel {i}" for i in range(len(metadata))]
        return channel_names

    @staticmethod
    def extract_channel_names_qptiff(path: Union[str, pathlib.Path]) -> list[str]:
        """
        Extract channel names from a QPTIFF file.
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
            channel_names = (
                qptiff_metadata.loc[is_marker]
                .drop_duplicates(["id", "markerName"])["markerName"]
                .tolist()
            )
        return channel_names

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

    def slice_array(self, ymin: int, ymax: int, xmin: int, xmax: int) -> zarr.Array:
        """
        Slice a zarr array.
        """
        return self.zimg[ymin:ymax, xmin:xmax]

    def slice_dict(
        self, ymin: int, ymax: int, xmin: int, xmax: int
    ) -> dict[str, zarr.Array]:
        """
        Slice a dictionary of zarr arrays.
        """
        return {
            channel_name: self.zimg_dict[channel_name][ymin:ymax, xmin:xmax]
            for channel_name in self.channel_names
        }


################################################################################
# Pyramidal OME-TIFF Writer
# https://github.com/labsyspharm/ome-tiff-pyramid-tools/blob/master/pyramid_assemble.py
################################################################################


class PyramidWriter:
    def __init__(
        self,
        in_imgs: list[zarr.Array],
        in_chns: list[str],
        target_shape: tuple[int, int],
        target_dtype: np.dtype,
    ):
        """
        Initialize the TiffWriter.

        Parameters
        ----------
        in_imgs : list[zarr.Array]
            List of zarr arrays.
        in_chns : list[str]
            List of channel names.
        target_shape : tuple[int, int]
            Target shape of the images.
        target_dtype : np.dtype
            Target data type of the images.
        """
        self.in_imgs = [img.astype(target_dtype) for img in in_imgs]
        self.in_chns = in_chns
        self.target_shape = target_shape
        self.target_dtype = target_dtype

    @classmethod
    def from_fs(
        cls,
        input_data: list[Union[str, pathlib.Path]],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """
        Process file paths input format.

        Parameters
        ----------
        input_data : list[Union[str, pathlib.Path]]
            A list of file paths to TIFF images
        channel_names : list[str], optional
            Names of the channels.
        is_mask : bool, optional
            Whether the images are masks.

        Returns
        -------
        PyramidWriter
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
                PyramidWriter._validate_image_2d(
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
                    PyramidWriter._validate_image_2d(
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

        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_array(
        cls,
        input_data: Union[np.ndarray, zarr.Array],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """
        Process array input format (2D or 3D).

        Parameters
        ----------
        input_data : Union[np.ndarray, zarr.Array]
            2D (will be treated as single channel) or 3D (C, H, W) array or zarr array.
        channel_names : list[str], optional
            Names of the channels.
        is_mask : bool, optional
            Whether the images are masks.

        Returns
        -------
        PyramidWriter
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
            PyramidWriter._validate_image_2d(
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

        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_dict(
        cls,
        input_data: dict[str, Union[np.ndarray, zarr.Array]],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
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
        PyramidWriter
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
                    PyramidWriter._validate_image_2d(
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
                        PyramidWriter._validate_image_2d(
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

        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @staticmethod
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

    @staticmethod
    def _create_metadata(pixel_size: float, channel_names: list[str]) -> dict:
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

    @staticmethod
    def _create_tile_generators(
        in_imgs: list[zarr.Array],
        output_f: pathlib.Path,
        num_channels: int,
        tile_size: int,
        is_mask: bool,
        target_dtype: np.dtype,
        target_shape: tuple[int, int],
        num_threads: int,
    ) -> tuple[callable, callable]:
        """Create tile generators for base and pyramid levels.

        Parameters
        ----------
        in_imgs : list[zarr.Array]
            List of input images as zarr arrays.
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
        target_shape : tuple[int, int]
            Target shape for the output image.

        Returns
        -------
        tuple[callable, callable]
            A tuple containing two generator functions:
            - First function generates tiles for the base level
            - Second function generates tiles for pyramid levels
        """
        # Calculate pyramid levels and shapes
        num_levels = max(1, int(np.ceil(np.log2(max(target_shape) / tile_size)) + 1))
        factors = 2 ** np.arange(num_levels)
        shapes = np.ceil(np.array(target_shape) / factors[:, None]).astype(int)
        cshapes = np.ceil(shapes / tile_size).astype(int)  # shapes of tiles

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

        return tiles0, tiles, num_levels, shapes, cshapes

    def export_ometiff_pyramid(
        self,
        output_f: Union[str, pathlib.Path],
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
        output_f : Union[str, pathlib.Path]
            Path to the output OME-TIFF file.
        pixel_size : float, optional
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

        # Create metadata
        metadata = PyramidWriter._create_metadata(
            pixel_size=pixel_size,
            channel_names=self.in_chns,
        )
        num_channels = len(self.in_chns)

        # Create tile generators
        (
            tiles0,
            tiles,
            num_levels,
            shapes,
            cshapes,
        ) = PyramidWriter._create_tile_generators(
            in_imgs=self.in_imgs,
            output_f=output_f,
            num_channels=num_channels,
            tile_size=tile_size,
            is_mask=is_mask,
            target_dtype=self.target_dtype,
            target_shape=self.target_shape,
            num_threads=num_threads,
        )

        # Write pyramid
        pbar = tqdm(
            total=sum(tile_shape[0] * tile_shape[1] for tile_shape in cshapes),
            desc="Writing tiles",
            bar_format=TQDM_FORMAT,
        )
        with tifffile.TiffWriter(output_f, ome=True, bigtiff=True) as writer:
            for level, shape in enumerate(shapes):
                if level == 0:
                    writer.write(
                        data=tiles0(),
                        shape=(num_channels,) + tuple(shape),
                        subifds=num_levels - 1,
                        dtype=self.target_dtype,
                        tile=(tile_size, tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                        metadata=metadata,
                    )
                    pbar.update(cshapes[level][0] * cshapes[level][1])
                else:
                    writer.write(
                        data=tiles(level),
                        shape=(num_channels,) + tuple(shape),
                        subfiletype=1,
                        dtype=self.target_dtype,
                        tile=(tile_size, tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                    )
                    pbar.update(cshapes[level][0] * cshapes[level][1])
        pbar.close()


# %%
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

    tiff_reader = TiffZarrReader.from_ometiff(ometiff_f)
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
    tiff_reader = TiffZarrReader.from_ometiff(ometiff_f)
    print(tiff_reader.channel_names)
    channels = ["DAPI", "CD45"]
    img = tiff_reader.zimg[tiff_reader.channel_index(channels), :100, :100]
    print(img.shape)

    img_dict = {
        channel_name: tiff_reader.zimg_dict[channel_name][:100, :100]
        for channel_name in channels
    }
    print({channel_name: img.shape for channel_name, img in img_dict.items()})

    # Test 3: write pyramid
    ometiff_f = Path(
        "/mnt/nfs/home/wenruiwu/projects/pyqupath/data/ometiff/test_3d_pyramid.ome.tiff"
    )
    tiff = TiffZarrReader.from_ometiff(ometiff_f)
    writer = PyramidWriter.from_array(tiff.zimg)
    writer.export_ometiff_pyramid(
        output_f=Path(
            "/mnt/nfs/home/wenruiwu/projects/pyqupath/data/ometiff/test_3d_pyramid_new.ome.tiff"
        ),
        tile_size=256,
    )


if __name__ == "__main__":
    main()
