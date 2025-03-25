# %%
"""Color utilities for pyqupath."""

import colorsys
from typing import List, Tuple, Union

import numpy as np


def generate_distinct_colors(
    n: int, saturation: float = 0.7, value: float = 0.95
) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct colors using HSV color space.

    This function generates colors by:
    1. Using the golden ratio to space hues evenly
    2. Using fixed saturation and value to ensure good visibility
    3. Avoiding similar hues when colors are adjacent

    Parameters
    ----------
    n : int
        Number of colors to generate
    saturation : float, optional
        Color saturation (0-1), default 0.7
    value : float, optional
        Color value/brightness (0-1), default 0.95

    Returns
    -------
    List[Tuple[int, int, int]]
        List of RGB color tuples with values 0-255
    """
    colors = []
    golden_ratio = 0.618033988749895  # Golden ratio conjugate

    # Start with primary colors for small n
    primary_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    if n <= len(primary_colors):
        return primary_colors[:n]

    # Use golden ratio method for larger n
    hue = 0
    for i in range(n):
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to 0-255 range
        rgb_int = tuple(int(255 * x) for x in rgb)
        colors.append(rgb_int)
        # Use golden ratio to space hues evenly
        hue = (hue + golden_ratio) % 1.0

    return colors


def assign_bright_colors(labels: List[Union[str, int]]) -> dict:
    """
    Assign bright RGB colors to variable values.

    Ensures good distinction between colors. Similar to QuPath's implementation
    but with improved color separation.

    Parameters
    ----------
    labels : List[Union[str, int]]
        List of labels that need distinct colors

    Returns
    -------
    dict
        Dictionary mapping each label to an RGB color tuple
    """
    n_colors = len(labels)
    colors = generate_distinct_colors(n_colors)
    return dict(zip(labels, colors))


def create_colormap(
    name: str, colors: List[Tuple[int, int, int]], n_interpolation: int = 256
) -> "DefaultColorMap":
    """
    Create a colormap from a list of RGB colors with interpolation.

    Parameters
    ----------
    name : str
        Name of the colormap
    colors : List[Tuple[int, int, int]]
        List of RGB color tuples (values 0-255)
    n_interpolation : int, optional
        Number of interpolation steps, default 256

    Returns
    -------
    DefaultColorMap
        A colormap object that can interpolate between the given colors
    """
    r = [c[0] for c in colors]
    g = [c[1] for c in colors]
    b = [c[2] for c in colors]
    return DefaultColorMap(name, r, g, b, n_interpolation)


class DefaultColorMap:
    """
    A color map that interpolates between given RGB values.

    This class provides a way to interpolate between a list of RGB colors to
    create a continuous color gradient. It pre-computes the interpolated colors
    for efficient lookup.
    """

    def __init__(
        self,
        name: str,
        r: List[int],
        g: List[int],
        b: List[int],
        n_colors: int = 256,
    ):
        """Initialize the color map with RGB values.

        Parameters
        ----------
        name : str
            Name of the colormap
        r : List[int]
            Red values (0-255)
        g : List[int]
            Green values (0-255)
        b : List[int]
            Blue values (0-255)
        n_colors : int, optional
            Number of interpolated colors, default 256
        """
        self.name = name
        self.r = np.array(r, dtype=np.int32)
        self.g = np.array(g, dtype=np.int32)
        self.b = np.array(b, dtype=np.int32)
        self.n_colors = n_colors

        # Pre-compute interpolated colors
        self._colors = {}
        self._precompute_colors()

    def _precompute_colors(self) -> None:
        """
        Pre-compute interpolated colors for faster lookup.

        This method pre-computes the interpolated colors for the entire range of
        the color map. It uses linear interpolation to compute the RGB values
        for each color index.
        """
        scale = (len(self.r) - 1) / self.n_colors

        for i in range(self.n_colors):
            ind = int(i * scale)
            residual = (i * scale) - ind

            # Linear interpolation for RGB values
            r = self.r[ind] + int((self.r[ind + 1] - self.r[ind]) * residual)
            g = self.g[ind] + int((self.g[ind + 1] - self.g[ind]) * residual)
            b = self.b[ind] + int((self.b[ind + 1] - self.b[ind]) * residual)

            self._colors[i] = self._pack_rgb(r, g, b)

        # Set the last color explicitly
        self._colors[self.n_colors - 1] = self._pack_rgb(
            self.r[-1], self.g[-1], self.b[-1]
        )

    @staticmethod
    def _pack_rgb(r: int, g: int, b: int) -> int:
        """Pack RGB values into a single 32-bit integer."""
        return (r << 16) | (g << 8) | b

    @staticmethod
    def _unpack_rgb(color: int) -> Tuple[int, int, int]:
        """Unpack a 32-bit integer into RGB values."""
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return r, g, b

    def get_color(self, value: float, min_value: float, max_value: float) -> int:
        """Get interpolated color for a value.

        Parameters
        ----------
        value : float
            The value to map to a color
        min_value : float
            Minimum value in the range
        max_value : float
            Maximum value in the range

        Returns
        -------
        int
            Packed RGB color value
        """
        ind = self._get_ind(value, min_value, max_value)
        return self._colors[ind]

    def _get_ind(self, value: float, min_value: float, max_value: float) -> int:
        """Convert a value to a color index."""
        max_val = max(min_value, max_value)
        min_val = min(min_value, max_value)

        if max_val == min_val:
            return 0

        ind = int(round((value - min_val) / (max_val - min_val) * (self.n_colors - 1)))
        ind = max(0, min(ind, self.n_colors - 1))

        return (self.n_colors - 1 - ind) if min_value > max_value else ind


# %%
# Examples
if __name__ == "__main__":
    # Example 1: Generate distinct colors
    print("\n1. Generating distinct colors:")
    colors5 = generate_distinct_colors(5)
    print("5 distinct colors:", colors5)

    colors10 = generate_distinct_colors(10)
    print("\n10 distinct colors:", colors10)

    # Example 2: Assign colors to labels
    print("\n2. Assigning colors to labels:")
    cell_types = ["Tumor", "Stroma", "Immune", "Necrosis", "Normal"]
    color_dict = assign_bright_colors(cell_types)
    print("Color assignments:")
    for label, color in color_dict.items():
        print(f"{label}: RGB{color}")

    # Example 3: Using colormaps
    print("\n3. Using colormaps:")

    # Red to Blue colormap
    red_blue_colors = [(255, 0, 0), (0, 0, 255)]
    red_blue_map = create_colormap("RedBlue", red_blue_colors)

    # Sample colors at different values
    values = [0, 0.25, 0.5, 0.75, 1.0]
    colors = [red_blue_map._unpack_rgb(red_blue_map.get_color(v, 0, 1)) for v in values]
    print("\nRed-Blue gradient at different values:")
    for value, color in zip(values, colors):
        print(f"Value {value:.2f}: RGB{color}")

    # Rainbow colormap
    rainbow_colors = [
        (255, 0, 0),  # Red
        (255, 165, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (148, 0, 211),  # Violet
    ]
    rainbow_map = create_colormap("Rainbow", rainbow_colors)
    rainbow_samples = [
        rainbow_map._unpack_rgb(rainbow_map.get_color(v, 0, 1))
        for v in np.linspace(0, 1, 10)
    ]
    print("\nRainbow gradient (10 samples):")
    for i, color in enumerate(rainbow_samples):
        print(f"Sample {i}: RGB{color}")
