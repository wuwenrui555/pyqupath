#!/usr/bin/env python3


def test_imports():
    """Test all module imports from pyqupath package."""
    try:
        from pyqupath import tiff

        print("✓ Successfully imported tiff module")
    except ImportError as e:
        print(f"✗ Failed to import tiff module: {e}")

    try:
        from pyqupath import ometiff

        print("✓ Successfully imported ometiff module")
    except ImportError as e:
        print(f"✗ Failed to import ometiff module: {e}")

    try:
        from pyqupath import constants

        print("✓ Successfully imported constants module")
    except ImportError as e:
        print(f"✗ Failed to import constants module: {e}")

    try:
        from pyqupath import geojson

        print("✓ Successfully imported geojson module")
    except ImportError as e:
        print(f"✗ Failed to import geojson module: {e}")

    try:
        from pyqupath import buffer

        print("✓ Successfully imported buffer module")
    except ImportError as e:
        print(f"✗ Failed to import buffer module: {e}")


if __name__ == "__main__":
    print("Testing imports from pyqupath package...")
    test_imports()
    print("\nImport test completed.")
