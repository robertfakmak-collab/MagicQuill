import torch
import pytest
from MagicQuill.magic_utils import get_bounding_box_from_mask, rgb_to_name

def test_get_bounding_box_empty_mask():
    # Test completely empty mask
    mask = torch.zeros((10, 10))
    assert get_bounding_box_from_mask(mask) == (0, 0, 0, 0)

def test_get_bounding_box_no_padding():
    # Test standard unpadded mask
    mask = torch.zeros((10, 10))
    mask[2:5, 3:7] = 1.0  # rows 2,3,4 (min 2, max 4), cols 3,4,5,6 (min 3, max 6)
    # top_left_x = 3/10 = 0.3
    # bottom_right_x = 6/10 = 0.6
    # top_left_y = 2/10 = 0.2
    # bottom_right_y = 4/10 = 0.4
    assert get_bounding_box_from_mask(mask, padded=False) == (0.3, 0.2, 0.6, 0.4)

def test_get_bounding_box_padded_width_less_than_height():
    # Test padded mask where width < height
    # Padded size will be max(6, 10) = 10
    # offset_x = (10 - 6) / 2 = 2
    # offset_y = 0
    mask = torch.zeros((10, 6))
    mask[2:5, 3:5] = 1.0  # rows 2,3,4 (min 2, max 4), cols 3,4 (min 3, max 4)
    # top_left_x = (3 + 2)/10 = 0.5
    # bottom_right_x = (4 + 2)/10 = 0.6
    # top_left_y = (2 + 0)/10 = 0.2
    # bottom_right_y = (4 + 0)/10 = 0.4
    assert get_bounding_box_from_mask(mask, padded=True) == (0.5, 0.2, 0.6, 0.4)

def test_get_bounding_box_padded_height_less_than_width():
    # Test padded mask where height < width
    # Padded size will be max(10, 6) = 10
    # offset_x = 0
    # offset_y = (10 - 6) / 2 = 2
    mask = torch.zeros((6, 10))
    mask[2:5, 3:5] = 1.0  # rows 2,3,4 (min 2, max 4), cols 3,4 (min 3, max 4)
    # top_left_x = (3 + 0)/10 = 0.3
    # bottom_right_x = (4 + 0)/10 = 0.4
    # top_left_y = (2 + 2)/10 = 0.4
    # bottom_right_y = (4 + 2)/10 = 0.6
    assert get_bounding_box_from_mask(mask, padded=True) == (0.3, 0.4, 0.4, 0.6)

def test_get_bounding_box_padded_square():
    # Test padded mask where height == width (square)
    # Padded size will be max(10, 10) = 10
    # offset_x = 0
    # offset_y = 0
    mask = torch.zeros((10, 10))
    mask[2:5, 3:7] = 1.0
    # Output should be the same as unpadded
    assert get_bounding_box_from_mask(mask, padded=True) == (0.3, 0.2, 0.6, 0.4)

def test_get_bounding_box_squeeze():
    # Test that mask with extra dimensions is squeezed correctly
    mask = torch.zeros((1, 1, 10, 10))
    mask[0, 0, 2:5, 3:7] = 1.0
    assert get_bounding_box_from_mask(mask, padded=False) == (0.3, 0.2, 0.6, 0.4)

def test_get_bounding_box_threshold_boundary():
    # Test values around 0.5 boundary
    mask = torch.zeros((10, 10))
    mask[2:5, 3:7] = 0.4  # Below threshold, shouldn't be counted
    assert get_bounding_box_from_mask(mask, padded=False) == (0, 0, 0, 0)

    mask[3:4, 4:5] = 0.5  # Exactly threshold, typically strict > is used
    assert get_bounding_box_from_mask(mask, padded=False) == (0, 0, 0, 0)

    mask[2:5, 3:7] = 0.6  # Above threshold, should be counted
    assert get_bounding_box_from_mask(mask, padded=False) == (0.3, 0.2, 0.6, 0.4)

def test_get_bounding_box_missing_rows_or_cols():
    # If all 1s are in a single column or single row
    mask = torch.zeros((10, 10))
    mask[5:6, 3:7] = 1.0  # Only row 5 is > 0.5
    # rows max = min = 5, cols min = 3, max = 6
    # top_left_y = 5/10 = 0.5, bottom_right_y = 5/10 = 0.5
    assert get_bounding_box_from_mask(mask, padded=False) == (0.3, 0.5, 0.6, 0.5)

    mask = torch.zeros((10, 10))
    mask[2:5, 5:6] = 1.0  # Only col 5 is > 0.5
    assert get_bounding_box_from_mask(mask, padded=False) == (0.5, 0.2, 0.5, 0.4)

def test_rgb_to_name_exact_match():
    # Test an exact match for a named color
    assert rgb_to_name((0, 0, 0)) == "black"
    assert rgb_to_name((255, 255, 255)) == "white"
    assert rgb_to_name((255, 0, 0)) == "red"

def test_rgb_to_name_closest_match():
    # Test an RGB color that doesn't exactly match any named web color
    # (255, 1, 1) is very close to red (255, 0, 0)
    # The rgb_to_name function expects tuples of elements that have a `.item()` method,
    # as used in closest_colour (e.g., requested_colour[0].item()). We will use PyTorch tensors.

    color1 = (torch.tensor(254), torch.tensor(255), torch.tensor(255))
    assert rgb_to_name(color1) == "white"

    color2 = (torch.tensor(255), torch.tensor(1), torch.tensor(1))
    assert rgb_to_name(color2) == "red"
