# %%
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import json
import numpy as np
import pathlib
import re
from pathlib import Path
from typing import Union
from xml.etree import ElementTree

import pandas as pd
import tifffile
import zarr
from pyqupath.geojson import load_geojson_to_gdf, polygon_to_mask


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

        # Get channel names
        self.channel_names = None
        if filetype == "ome.tiff":
            _extract_channels = self.extract_channels_from_ometiff
        elif filetype == "qptiff":
            _extract_channels = self.extract_channels_from_qptiff
        try:
            self.channel_names = _extract_channels(self.path)
        except Exception:
            pass

        # Initialize tifffile reader
        self.zimg = zarr.open(tifffile.imread(ometiff_f, level=0, aszarr=True))
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


geojson_f = Path(
    "/mnt/nfs/home/wenruiwu/projects/pyqupath/data/geojson/test_updated_2.geojson"
)
gdf = load_geojson_to_gdf(geojson_f)

for _, row in gdf.iterrows():
    polygon = row["geometry"]
    y_min, x_min, y_max, x_max = polygon.bounds

    # Get image dimensions from the tiff object
    ometiff_f = Path(
        "/mnt/nfs/home/wenruiwu/projects/pyqupath/data/ometiff/test_3d_pyramid.ome.tiff"
    )
    tiff_reader = TiffZarrReader(ometiff_f)
    height, width = tiff_reader.zimg.shape[-2:]

    # Crop to bounds first
    y_min = max(0, int(np.floor(y_min)))
    y_max = min(height, int(np.ceil(y_max)))
    x_min = max(0, int(np.floor(x_min)))
    x_max = min(width, int(np.ceil(x_max)))
    print(y_min, y_max, x_min, x_max)

    # Get the rectangular crop
    img = tiff_reader.zimg[:, y_min:y_max, x_min:x_max]

    # Extract coordinates and shift them
    y_coords = [y - y_min for y in polygon.exterior.coords.xy[0]]
    x_coords = [x - x_min for x in polygon.exterior.coords.xy[1]]
    shifted_polygon = Polygon(zip(x_coords, y_coords))
    mask = polygon_to_mask(shifted_polygon, (y_max - y_min, x_max - x_min))

    fill_value = 0
    img_masked = img * mask + fill_value * (1 - mask)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img[0], cmap="gray")
    axs[1].imshow(mask, cmap="gray")
    axs[2].imshow(img_masked[0], cmap="gray")
    plt.show()
    fig.tight_layout()


if __name__ == "__main__":
    main()

# %%
